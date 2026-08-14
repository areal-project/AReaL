# memrl/envs/webshop_env.py
"""WebShop environment wrapper for MemRL"""
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Local copy of WebShop environment
WEBSHOP_PATH = Path(__file__).parent / "webshop"
if str(WEBSHOP_PATH) not in sys.path:
    sys.path.insert(0, str(WEBSHOP_PATH))
    sys.path.insert(0, str(WEBSHOP_PATH / "web_agent_site"))


class WebShopEnv:
    """Wrapper for WebShop text environment"""

    def __init__(
        self,
        file_path: str = None,
        num_products: int = None,
        human_goals: bool = True,
        seed: int = 42,
        max_steps: int = 30,
        goal_list: Optional[List[dict]] = None,
        **kwargs
    ):
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

        default_file_path = str(WEBSHOP_PATH / "data" / "items_shuffle_1000.json")
        if not Path(default_file_path).exists():
            default_file_path = str(Path(__file__).parent.parent.parent / "data" / "webshop" / "items_shuffle_1000.json")
        self.file_path = file_path or default_file_path

        self.env = WebAgentTextEnv(
            observation_mode='text',
            file_path=self.file_path,
            num_products=num_products,
            human_goals=human_goals,
            seed=seed,
            **kwargs
        )

        # Inject filtered/reordered goals if provided
        if goal_list is not None:
            self._inject_goals(goal_list)

        self.current_goal = ""
        self.steps = 0
        self.max_steps = max_steps

    def _inject_goals(self, goal_list: List[dict]):
        """
        Replace SimServer.goals with a filtered subset matching goal_list.

        goal_list entries have {asin, instruction, instruction_attributes, instruction_options}.
        SimServer.goals entries have {asin, instruction_text, attributes, goal_options, ...}.

        Match by (asin, frozenset(attributes)) since instruction_text has price suffix appended.
        """
        server = self.env.server
        all_goals = server.goals

        # Build lookup: (asin, frozenset(attributes)) -> list of env goal dicts
        goal_lookup = {}
        for g in all_goals:
            key = (g['asin'], frozenset(g['attributes']))
            goal_lookup.setdefault(key, []).append(g)

        filtered_goals = []
        unmatched = 0
        for split_goal in goal_list:
            key = (split_goal['asin'], frozenset(split_goal['instruction_attributes']))
            candidates = goal_lookup.get(key, [])
            if candidates:
                # Pop first match to handle duplicates
                filtered_goals.append(candidates.pop(0))
            else:
                unmatched += 1

        if unmatched > 0:
            logger.warning(
                "[WebShopEnv] %d/%d goals could not be matched in env (products may be missing)",
                unmatched, len(goal_list),
            )

        if len(filtered_goals) != len(goal_list):
            raise ValueError(
                f"[WebShopEnv] Goal injection mismatch: requested={len(goal_list)}, "
                f"matched={len(filtered_goals)}, missing={len(goal_list) - len(filtered_goals)}. "
                f"Session indices would go out of range. Check that split_info goals match the product file."
            )

        server.goals = filtered_goals
        server.weights = [g.get('weight', 1) for g in filtered_goals]
        import numpy as np
        server.cum_weights = [0] + np.cumsum(server.weights).tolist()

        logger.info("[WebShopEnv] Injected %d goals (from %d requested)", len(filtered_goals), len(goal_list))

    def reset(self, session_idx: int = None) -> Tuple[str, Dict[str, Any]]:
        # NOTE: the session index MUST be passed into the underlying reset(). Setting
        # self.env.session beforehand does nothing — reset() overwrites it, and when
        # called with session=None the server picks a RANDOM goal (session_int=None ->
        # random_idx) and a random session string. Passing session=session_idx makes
        # session_int an int so the server loads self.goals[session_idx] (our split's
        # N-th injected goal). reset() also returns a (obs, None) tuple that must be
        # unpacked. See docs/experiments/webshop/2026-06-18_nomem_memrl_run_bugs.md.
        if session_idx is not None:
            n_goals = len(self.env.server.goals)
            if session_idx < 0 or session_idx >= n_goals:
                raise IndexError(
                    f"[WebShopEnv] session_idx {session_idx} out of range for {n_goals} goals"
                )
            obs, _ = self.env.reset(session=session_idx)
        else:
            obs, _ = self.env.reset()
        goal = ""
        try:
            session_id = str(self.env.session)
            session_data = self.env.server.user_sessions.get(session_id, {})
            goal_data = session_data.get('goal', {})
            goal = goal_data.get('instruction_text', '') if isinstance(goal_data, dict) else ''
        except Exception:
            pass
        self.current_goal = goal
        self.steps = 0

        return obs, {'goal': goal, 'session': getattr(self.env, 'session', session_idx)}

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.steps += 1
        obs, reward, done, info = self.env.step(action)

        if self.steps >= self.max_steps and not done:
            done = True

        return obs, reward, done, {
            'steps': self.steps,
            'goal': self.current_goal,
            'reward': reward,
        }

    def get_available_actions(self) -> Dict:
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        return {}

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


def get_webshop_sessions(
    split_info_path: str,
    split: str = "train",
) -> Tuple[List[int], List[dict]]:
    """
    Load session indices and goal metadata for the given split from split_info.json.

    Args:
        split_info_path: Path to split_info.json
        split: One of "train", "val", "ood" (also accepts legacy names)

    Returns:
        (session_indices, goals_list) where session_indices are 0..N-1
        and goals_list contains {asin, instruction, instruction_attributes, instruction_options}.
    """
    with open(split_info_path) as f:
        split_info = json.load(f)

    # Map legacy split names
    split_key_map = {
        "train": "train",
        "val": "val",
        "ood": "ood",
        "eval_in_distribution": "val",
        "eval_out_of_distribution": "ood",
    }
    split_key = split_key_map.get(split, split)

    if split_key not in split_info:
        raise ValueError(f"Unknown split '{split}'. Available: {list(split_info.keys())}")

    goals = split_info[split_key]["goals"]
    indices = list(range(len(goals)))
    return indices, goals
