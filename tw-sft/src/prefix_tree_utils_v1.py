import trl
import torch
import re
import time
from typing import List, Any, Dict, Union, Optional, Tuple
from collections import defaultdict

print("Imported prefix tree collator v1")

TAG_TOKEN_IDS = {
    'parallel_start': '<Parallel>',
    'parallel_end': '</Parallel>',
    'thread_start': '<Thread>',
    'thread_end': '</Thread>',
    'outlines_start': '<Outlines>',
    'outlines_end': '</Outlines>',
    'trial_start': '<Trial>',
    'trial_end': '</Trial>',
    'subtask_start': '<Subtask>',
    'subtask_end': '</Subtask>',
    'conclusion_start': '<Conclusion>',
    'conclusion_end': '</Conclusion>',
}

def _get_single_token_id(tokenizer, token: str) -> Optional[int]:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return token_ids[0]

def _find_closing_tag_pos_tokenized(
    token_ids: List[int], open_id: int, close_id: int, start_idx: int
) -> int:
    """Finds the index of the matching closing tag ID, handling nesting of the same tag."""
    level = 1
    for i in range(start_idx + 1, len(token_ids)):
        if token_ids[i] == open_id:
            level += 1
        elif token_ids[i] == close_id:
            level -= 1
        if level == 0:
            return i
    raise ValueError(f"No matching closing tag ID '{close_id}' for opening ID '{open_id}' found.")

def _get_direct_children_tokenized(token_ids: List[int], START_ID_TO_TAG_INFO: Dict[int, Tuple[str, int]]) -> List[Dict]:
    """
    Returns a list of dicts for each direct child element by correctly
    handling nested structures.
    """
    children: List[Dict] = []
    i = 0
    while i < len(token_ids):
        token_id = token_ids[i]

        # Check if the current token is a known start tag
        if token_id in START_ID_TO_TAG_INFO:
            tag_name, end_id = START_ID_TO_TAG_INFO[token_id]
            start_idx = i

            # Find the matching closing tag, correctly handling all nested content
            try:
                end_idx = _find_closing_tag_pos_tokenized(
                    token_ids, open_id=token_id, close_id=end_id, start_idx=start_idx
                )
            except ValueError as e:
                raise ValueError(f"Malformed token sequence: {e}")

            # This is a complete, top-level child. Add it to our list.
            children.append({
                'tag': tag_name,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'content_ids': token_ids[start_idx : end_idx + 1]
            })

            # Jump the cursor past this entire child element to find the next sibling
            i = end_idx + 1
        else:
            # Not a start tag, just move to the next token
            i += 1
            
    return children

def generate_seq_list_tokenized(
        input_ids: List[int], 
        START_ID_TO_TAG_INFO: Dict[int, Tuple[str, int]], 
        parallel_start_id: int,
        parallel_end_id: int
    ) -> List[List[int]]:
    """
    Splits an input containing one outer <Parallel>…</Parallel> into:
      - one sequence per repeated <Thread>…</Thread> (keeping head static + that one)
      - one final sequence: head + all <Thread>…</Thread> + tail + closing + post
    Returns the **final combined/original** sequence as the FIRST element,
    followed by the per-branch prefix sequences.
    """
    try:
        open_idx = input_ids.index(parallel_start_id)
        close_idx = _find_closing_tag_pos_tokenized(
            input_ids, parallel_start_id, parallel_end_id, open_idx
        )
    except ValueError:
        # No <Parallel>... so the only sequence is the original
        return [input_ids]

    pre_ids   = input_ids[:open_idx]
    inner_ids = input_ids[open_idx + 1:close_idx]
    post_ids  = input_ids[close_idx + 1:]

    children = _get_direct_children_tokenized(inner_ids, START_ID_TO_TAG_INFO)
    if not children:
        return [input_ids]

    counts: Dict[str, int] = {}
    for ch in children:
        counts[ch['tag']] = counts.get(ch['tag'], 0) + 1
    
    branch_tag = next((t for t, c in counts.items() if c > 1), 'Thread')
    branch_ids_list = [ch['content_ids'] for ch in children if ch['tag'] == branch_tag]

    if not branch_ids_list:
        return [input_ids]

    first_branch_child_idx = next((i for i, ch in enumerate(children) if ch['tag'] == branch_tag), -1)
    last_branch_child_idx  = max((i for i, ch in enumerate(children) if ch['tag'] == branch_tag), default=-1)
    
    head_ids = inner_ids[:children[first_branch_child_idx]['start_idx']]
    tail_ids = inner_ids[children[last_branch_child_idx]['end_idx'] + 1:]

    seqs: List[List[int]] = []

    # ---- Build the combined/original sequence FIRST ----
    post_seq_list = generate_seq_list_tokenized(
        post_ids,
        START_ID_TO_TAG_INFO,
        parallel_start_id=parallel_start_id,
        parallel_end_id=parallel_end_id
    )
    all_branches_flat = [item for sublist in branch_ids_list for item in sublist]
    for post_seq in post_seq_list:
        combined = (
            pre_ids + [parallel_start_id] + head_ids + all_branches_flat + tail_ids +
            [parallel_end_id] + post_seq
        )
        seqs.append(combined)

    # ---- Then add one prefix-style sequence per branch ----
    for b_ids in branch_ids_list:
        seqs.append(pre_ids + [parallel_start_id] + head_ids + b_ids)

    return seqs

def process_input_ids(trajectories, tokenizer, max_length: Optional[int] = None):
    """
    Build a trie in first-seen (insertion) order and create a tree-ancestry attention mask.
    Children are traversed in the order they were first inserted, which makes the
    original trajectory a prefix of the merged traversal.
    """
    if len(trajectories) > 1:
        assert len(trajectories[0]) > max([len(traj) for traj in trajectories[1:]]), f"Expected trajectories to be longer than the longest trajectory, got {len(trajectories[0])} vs max({[len(traj) for traj in trajectories]})"

    # 1) Build a tiny trie (defaultdict preserves insertion order on keys)
    Trie = lambda: defaultdict(Trie)
    root = Trie()
    for traj in trajectories:
        node = root
        for tok in traj:
            node = node[tok]

    # 2) Flatten with an explicit stack, recording tin/tout in insertion order
    flat_ids: List[int] = []
    position_ids: List[int] = []
    parent_pointers: List[int] = []
    tin: List[Optional[int]] = []
    tout: List[Optional[int]] = []
    timer = 0

    # Stack entries: (node, depth, parent_idx, children_iter, my_idx)
    stack = [(root, 0, -1, iter(root.items()), None)]
    while stack:
        node, depth, parent_idx, children, my_idx = stack[-1]

        if my_idx is not None and tin[my_idx] is None:
            tin[my_idx] = timer
            timer += 1

        try:
            token, child = next(children)
        except StopIteration:
            if my_idx is not None:
                tout[my_idx] = timer
                timer += 1
            stack.pop()
        else:
            idx = len(flat_ids)
            flat_ids.append(token)
            position_ids.append(depth)
            parent_pointers.append(parent_idx)
            tin.append(None)
            tout.append(None)
            # descend using insertion order
            stack.append((child, depth+1, idx, iter(child.items()), idx))

    # 3) Tensors
    input_ids    = torch.tensor(flat_ids,      dtype=torch.long)
    position_ids = torch.tensor(position_ids,  dtype=torch.long)
    tin_t        = torch.tensor(tin,           dtype=torch.long)
    tout_t       = torch.tensor(tout,          dtype=torch.long)

    # 4) Vectorized ancestor mask via entry/exit times
    tin_row  = tin_t.unsqueeze(0)   # (1, N)
    tin_col  = tin_t.unsqueeze(1)   # (N, 1)
    tout_row = tout_t.unsqueeze(0)  # (1, N)

    bool_attention_mask = (tin_row <= tin_col) & (tout_row >= tin_col)

    attention_mask = torch.full_like(bool_attention_mask, -torch.inf, dtype=torch.float)
    attention_mask = attention_mask.masked_fill(bool_attention_mask, 0.0)

    if max_length is not None:
        # Truncate to max_length if specified
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length, :max_length]
        position_ids = position_ids[:max_length]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
    }


class PrefixTreeDataCollatorForCompletionOnlyLM(trl.DataCollatorForCompletionOnlyLM):
    def __init__(self, *args, max_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length

        # Define tags and get their corresponding token IDs
        TAG_TOKEN_ALIASES = {
            'Parallel': [('<Parallel>', '</Parallel>')],
            'Thread': [('<Thread>', '</Thread>')],
            'Outlines': [('<Outlines>', '</Outlines>')],
            'Outline': [('<Outline>', '</Outline>'), ('<Trial>', '</Trial>'), ('<Subtask>', '</Subtask>')],
            'Conclusion': [('<Conclusion>', '</Conclusion>')],
        }
        tag_ids: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for name, aliases in TAG_TOKEN_ALIASES.items():
            for start_token, end_token in aliases:
                start_id = _get_single_token_id(self.tokenizer, start_token)
                end_id = _get_single_token_id(self.tokenizer, end_token)
                if start_id is not None and end_id is not None:
                    tag_ids[name].append((start_id, end_id))

        assert tag_ids['Parallel'], "Parallel start/end token IDs must be defined."
        assert tag_ids['Thread'], "Thread start/end token IDs must be defined."

        # Inverse mapping from a start token ID to its tag name and end ID
        self.START_ID_TO_TAG_INFO = {}
        for name, id_pairs in tag_ids.items():
            for start_id, end_id in id_pairs:
                self.START_ID_TO_TAG_INFO[start_id] = (name, end_id)

        self.parallel_start_id, self.parallel_end_id = tag_ids['Parallel'][0]
        self.thread_start_id, self.thread_end_id = tag_ids['Thread'][0]

        # print(f"Thread Start ID: {self.thread_start_id}, Thread End ID: {self.thread_end_id}")
        # print(f"Parallel Start ID: {self.parallel_start_id}, Parallel End ID: {self.parallel_end_id}")

    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]], profile: bool = False) -> Dict[str, Any]:
        if profile:
            t1_torch_call = time.time()
        # First, generate full attention masks and position ids for complete sequences
        input_ids_all = []
        attention_masks_all = []
        position_ids_all = []

        examples_processed = []
        
        for example in examples:
            # Get the complete input_ids (before any truncation)
            assert isinstance(example, dict)
            input_ids = example['input_ids']
            # print(example.keys())

            if profile:
                t1 = time.time()
            trajectories = generate_seq_list_tokenized(
                input_ids,
                START_ID_TO_TAG_INFO=self.START_ID_TO_TAG_INFO,
                parallel_start_id=self.parallel_start_id,
                parallel_end_id=self.parallel_end_id
            )
            if profile:
                print(f"Generated the sequence list in {time.time() - t1:.4f} seconds.")

            # sort trajectories by lengths (longest first)
            # trajectories.sort(key=len, reverse=True)

            # print("**Num traj:**", len(trajectories))
            # print("**Trajectory:**", self.tokenizer.decode(trajectories[0]))

            if profile:
                t1 = time.time()
            trajectories = process_input_ids(trajectories, tokenizer=self.tokenizer, max_length=self.max_length)
            if profile:
                print(f"Processed the trajectories in {time.time() - t1:.4f} seconds.")

            # Generate full attention mask and position ids based on complete sequence
            input_ids = trajectories['input_ids']
            attention_mask = trajectories['attention_mask']
            position_ids = trajectories['position_ids']

            input_ids_all.append(input_ids)
            attention_masks_all.append(attention_mask)
            position_ids_all.append(position_ids)

            examples_processed.append({
                'input_ids': input_ids,
                # 'attention_mask': attention_mask,
                # 'position_ids': position_ids
            })

        if profile:
            print(f"torch_call before super torch_call: {time.time() - t1_torch_call:.4f} seconds.")
        # Apply the standard collation with truncated examples
        batch = super().torch_call(examples_processed)

        if profile:
            print(f"torch_call after super torch_call: {time.time() - t1_torch_call:.4f} seconds.")

        # Get the final sequence length after truncation
        final_seq_len = batch['input_ids'].shape[1]

        assert (self.max_length is None) or (final_seq_len <= self.max_length), \
            f"Final sequence length {final_seq_len} exceeds max_length {self.max_length}. " \
            "This should not happen as we truncate in the collator."
        
        # Create custom attention masks and position ids with the same truncation
        batch['attention_mask'] = torch.zeros(len(examples), 1, final_seq_len, final_seq_len, dtype=torch.float, device='cpu')
        batch['position_ids'] = torch.zeros(len(examples), final_seq_len, dtype=torch.long, device='cpu')

        for i in range(len(examples)):
            # Apply the same truncation to attention mask and position ids
            batch['attention_mask'][i, 0] = attention_masks_all[i][:final_seq_len, :final_seq_len]
            batch['position_ids'][i] = position_ids_all[i][:final_seq_len]
            # Should pass:
            # assert torch.all(batch['input_ids'][i] == input_ids_all[i][:final_seq_len])
        
        if profile:
            print(f"torch_call completed in {time.time() - t1_torch_call:.4f} seconds.")

        print(f"Final batch size: {len(examples)}, sequence length: {final_seq_len}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Position ids shape: {batch['position_ids'].shape}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        return batch
