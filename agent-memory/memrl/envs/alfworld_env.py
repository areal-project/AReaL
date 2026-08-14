import logging
import os
import tempfile
import types
from functools import partial
from .base import IEnv
import yaml
from typing import Optional

logger = logging.getLogger(__name__)

# ALFWorld is an optional dependency. Import lazily so that users can run
# non-ALFWorld benchmarks without installing it.
try:
    from alfworld.agents.environment import get_environment  # type: ignore

    # fast_downward.load_lib() copies libdownward.so into a TemporaryDirectory
    # for every TextWorld environment. On CPFS, directory entry propagation can
    # briefly make shutil.rmtree() report ENOTEMPTY after the library loaded.
    # Limit the workaround to fast_downward's private temporary directories; do
    # not change tempfile.TemporaryDirectory globally.
    try:
        import fast_downward.interface as _fd_interface  # type: ignore

        _fd_interface.tempfile = types.SimpleNamespace(
            TemporaryDirectory=partial(
                tempfile.TemporaryDirectory, ignore_cleanup_errors=True
            )
        )
    except (ImportError, AttributeError):
        pass
except ModuleNotFoundError as _e:
    get_environment = None  # type: ignore[assignment]
    _ALFWORLD_IMPORT_ERROR = _e

def load_config_from_path(config_path: str, params=None):
    assert os.path.exists(config_path), f"Invalid config file: {config_path}"
    with open(config_path) as reader:
        config = yaml.safe_load(reader)
    if params is not None:
        for param in params:
            fqn_key, value = param.split("=")
            entry_to_change = config
            keys = fqn_key.split(".")
            for k in keys[:-1]:
                entry_to_change = entry_to_change[k]
            entry_to_change[keys[-1]] = value
    return config


class AlfWorldEnv(IEnv):
    """
    ALFWorld wrapper environment for training memory agents with support for batch_size > 1.
    """

    def __init__(self, config_path: str, preconfigured_env: Optional[object] = None, 
                 task_type: str = "train", batch_size: int = 1):
        """
        Args:
            config_path (str): Path to the main ALFWorld config YAML.
            preconfigured_env (Optional[object]): If provided, the wrapper will use this
                                                   environment instance instead of creating its own.
            task_type (str): The dataset split to use if creating a default env.
            batch_size (int): The batch size to use if creating a default env.
        """
        self.config_path = config_path
        self.batch_size = batch_size
        self.current_trace_list = [[] for _ in range(batch_size)]
        
        if preconfigured_env:
            # If a specific environment is passed in, use it directly.
            self.env = preconfigured_env
        else:
            # Otherwise, create a default environment that samples from the whole split.
            if get_environment is None:
                raise ModuleNotFoundError(
                    "ALFWorld is not installed. Install ALFWorld dependencies to run the ALFWorld benchmark."
                ) from _ALFWORLD_IMPORT_ERROR
            config = load_config_from_path(config_path)
            env_type = config['env']['type']
            underlying_env_controller = get_environment(env_type)(config, train_eval=task_type)
            self.env = underlying_env_controller.init_env(batch_size=batch_size)

    def reset(self):
        """
        Reset environments and return first observations.
        
        Returns:
            list[dict]: [{"obs": str, "info": dict}, ...] (length=batch_size)
        """
        obs_list, info = self.env.reset()
        self.initial_obs = obs_list
        self.current_trace_list = [[] for _ in range(self.batch_size)]

        results = []
        for i, obs in enumerate(obs_list):
            per_env_info = {k: v[i] for k, v in info.items()}
            results.append({
                "obs": self._process_obs(obs),
                "info": per_env_info
            })
        return results



    def step(self, actions: list):
        """
        Execute one step in the environments.

        Args:
            actions (list[str]): textual actions for each environment

        Returns:
            list[dict]: [{"obs": str, "reward": float, "done": bool, "info": dict}, ...]
        """
        assert isinstance(actions, list), "When batch_size>1, actions must be a list"

        try:
            obs_list, reward_list, done_list, infos = self.env.step(actions)
            if any(done_list):
                logger.info(f"[ENV DEBUG] done_list={list(done_list)}, reward_list={list(reward_list)}, won={infos.get('won', 'N/A')}")
        except Exception as e:
            # If any environment process crashes (e.g., EOFError from workers),
            # mark ALL envs as done with error=True instead of silently returning
            # non-terminal zero-reward transitions. The previous behavior fake-
            # extended trajectories with empty obs for the rest of max_steps and
            # silently logged them as success=False, biasing SR downward and
            # hiding infra faults. Callers treat info.error as aborted episode
            # (done=True, success=False, aborted=True).
            logger.error("ALFWorld env.step failed; marking all slots aborted. Error: %s", e, exc_info=True)
            results = []
            err_msg = str(e)
            for i, action in enumerate(actions):
                step_data = {
                    "obs": "",
                    "reward": 0.0,
                    "done": True,
                    "info": {"error": err_msg},
                }
                self.current_trace_list[i].append({"action": action, **step_data})
                results.append(step_data)
            return results

        results = []
        for i, (obs, reward, done, action) in enumerate(zip(obs_list, reward_list, done_list, actions)):
            info_i = {k: v[i] for k, v in infos.items()}

            step_data = {
                "obs": self._process_obs(obs),
                "reward": reward,
                "done": done,
                "info": info_i
            }
            self.current_trace_list[i].append({"action": action, **step_data})
            results.append(step_data)
        return results


    def current_trace(self, env_id=None):
        """Return full trajectory for each env (or one env if env_id provided)."""
        if env_id is None:
            return self.current_trace_list
        return self.current_trace_list[env_id]

    def close(self):
        """
        Close ALFWorld env and unregister its textworld gym env_id so the
        process-global gym registry doesn't accumulate one entry per mini-batch
        (textworld.gym.register_games stores wrappers and config forever).

        If the underlying TextWorld pipes are already broken, swallow the
        exception so the caller can safely discard this env and move on.

        Returns:
            bool: True if closed cleanly, False if an error occurred.
        """
        closed_ok = True
        try:
            self.env.close()
        except Exception as e:
            logger.warning("ALFWorld env.close failed; discarding this env. Error: %s", e, exc_info=True)
            closed_ok = False

        # Unregister textworld env_id from gym's process-global registry. The
        # env_id is set by the runner's envs_built() right after register_games.
        # See HIGH #2 in code review: without this, 77 batches = 77 registry
        # entries + wrappers retained for the entire process lifetime.
        env_id = getattr(self, "_textworld_env_id", None)
        if env_id:
            # Try the modern gymnasium API first (used by current alfworld+textworld
            # singularity image), then fall back to legacy gym. Either way, log at
            # debug not warning — registry growth is a slow leak, not a hot crash.
            registry = None
            try:
                import gymnasium  # type: ignore
                registry = getattr(gymnasium.envs.registration, "registry", None)
            except Exception:
                try:
                    import gym  # type: ignore
                    registry = getattr(gym.envs.registration, "registry", None)
                except Exception:
                    pass
            if registry is not None:
                try:
                    # Newer gym/gymnasium: registry is a dict-like; older: has env_specs attr.
                    if hasattr(registry, "env_specs"):
                        registry.env_specs.pop(env_id, None)
                    elif hasattr(registry, "pop"):
                        registry.pop(env_id, None)
                except Exception as e:
                    logger.debug("Failed to unregister textworld env_id %s: %s", env_id, e)
        return closed_ok

    def _process_obs(self, obs):
        """
        Convert ALFWorld raw observation into a format suitable for memory agent.
        Default: return plain text.
        """
        if isinstance(obs, list):
            return " ".join(obs)
        return str(obs)
