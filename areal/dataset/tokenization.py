# SPDX-License-Identifier: Apache-2.0


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def get_multimodal_sft_loss_mask(
    input_ids,
    prompt: str,
    sequence: str,
    tokenizer,
) -> list[int]:
    """Build an SFT mask when a processor may expand prompt-side image tokens."""
    prompt_token = tokenizer.encode(prompt)
    sequence_token = tokenizer.encode(sequence)
    prompt_length = _common_prefix_length(prompt_token, sequence_token)

    # Multimodal processors may expand an image placeholder into several tokens.
    # Geometry3K and CLEVR keep every image in the prompt, so the resulting length
    # difference belongs entirely to the masked prompt prefix.
    prompt_length += len(input_ids) - len(sequence_token)
    if not 0 <= prompt_length <= len(input_ids):
        raise ValueError(
            "Cannot align the tokenized prompt with the processed input sequence"
        )

    return [0] * prompt_length + [1] * (len(input_ids) - prompt_length)
