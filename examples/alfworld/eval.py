"""ALFWorld Eval Script — evaluate a checkpoint on train/val/ood splits.

Serves the model with vLLM, runs episodes via env_server, reports SR per split.

Usage:
    python examples/alfworld/eval.py \
        --model_path /path/to/checkpoint \
        --splits train valid_seen valid_unseen \
        --max_episodes 200 \
        --max_steps 30
"""
import argparse
import asyncio
import json
import glob
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval")

sys.path.append(str(Path(__file__).parent.parent.parent))
from examples.alfworld.workflow import parse_action, SYSTEM_PROMPT
from examples.alfworld.env_client import EnvClient


def load_game_files(data_root: str, split: str) -> list[str]:
    """Load solvable game files for a split."""
    split_dir = os.path.join(data_root, split)
    game_files = sorted(glob.glob(os.path.join(split_dir, "**", "*.tw-pddl"), recursive=True))
    solvable = []
    for gf in game_files:
        try:
            with open(gf) as f:
                data = json.load(f)
            if data.get("solvable", True):
                solvable.append(gf)
        except Exception:
            solvable.append(gf)
    logger.info(f"Split {split}: {len(solvable)} solvable / {len(game_files)} total")
    return solvable


async def run_episode(env: EnvClient, game_file: str, openai_client, model: str, max_steps: int) -> dict:
    """Run a single episode. Returns dict with success, steps, game_file, trajectory."""
    try:
        env_id = await env.create(game_file)
    except Exception as e:
        return {"game_file": game_file, "success": False, "steps": 0, "error": f"create: {e}"}

    try:
        obs, info = await env.reset(env_id)
    except Exception as e:
        await env.close(env_id)
        return {"game_file": game_file, "success": False, "steps": 0, "error": f"reset: {e}"}

    task_description = "\n".join(obs.split("\n\n")[1:]) if "\n\n" in obs else obs
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Now, it's your turn to solve a new task.\n{task_description}"},
    ]
    trajectory = []

    try:
        for step in range(max_steps):
            try:
                response = await openai_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    max_tokens=256,
                    temperature=0.6,
                )
            except Exception as e:
                return {"game_file": game_file, "success": False, "steps": step, "error": f"llm: {e}", "trajectory": trajectory}

            content = response.choices[0].message.content if response.choices else ""
            action = parse_action(content)
            messages.append({"role": "assistant", "content": content})
            trajectory.append({"action": action, "thought": content})

            try:
                obs, reward, done, won = await env.step(env_id, action)
            except Exception:
                return {"game_file": game_file, "success": False, "steps": step + 1, "error": "step_failed", "trajectory": trajectory}

            trajectory[-1]["obs"] = obs
            trajectory[-1]["done"] = done

            if done:
                success = won or reward > 0
                return {"game_file": game_file, "success": success, "steps": step + 1, "trajectory": trajectory}

            messages.append({"role": "user", "content": f"Observation: {obs.strip()}"})

        return {"game_file": game_file, "success": False, "steps": max_steps, "trajectory": trajectory}
    finally:
        await env.close(env_id)


async def eval_split(
    game_files: list[str],
    env: EnvClient,
    openai_client,
    model: str,
    max_steps: int,
    max_episodes: int,
    concurrency: int = 16,
) -> list[dict]:
    """Evaluate a split with bounded concurrency."""
    import random
    files = game_files[:max_episodes] if max_episodes < len(game_files) else game_files
    random.shuffle(files)

    sem = asyncio.Semaphore(concurrency)
    results = []

    async def run_one(gf):
        async with sem:
            return await run_episode(env, gf, openai_client, model, max_steps)

    tasks = [run_one(gf) for gf in files]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 20 == 0:
            successes = sum(1 for r in results if r["success"])
            logger.info(f"  Progress: {i+1}/{len(files)}, SR so far: {successes}/{i+1} = {successes/(i+1):.3f}")

    return results


async def main_async(args):
    from openai import AsyncOpenAI

    openai_client = AsyncOpenAI(
        base_url=f"http://127.0.0.1:{args.vllm_port}/v1",
        api_key="dummy",
    )
    env = EnvClient(base_url=f"http://127.0.0.1:{args.env_port}")

    all_results = {}
    for split in args.splits:
        logger.info(f"=== Evaluating split: {split} ===")
        game_files = load_game_files(args.data_root, split)
        if not game_files:
            logger.warning(f"No game files for split {split}, skipping")
            continue

        results = await eval_split(
            game_files, env, openai_client, "default",
            max_steps=args.max_steps,
            max_episodes=args.max_episodes,
            concurrency=args.concurrency,
        )

        successes = sum(1 for r in results if r["success"])
        errors = sum(1 for r in results if r.get("error"))
        valid = len(results) - errors
        sr = successes / valid if valid > 0 else 0
        logger.info(f"  Split {split}: SR = {successes}/{valid} = {sr:.4f} ({errors} errors excluded)")
        all_results[split] = {"sr": sr, "successes": successes, "valid": valid, "errors": errors, "total": len(results)}

        # Save per-episode details (trajectories only for failures to save space)
        if args.save_trajectories:
            traj_file = (args.output or "/tmp/areal/eval_results.json").replace(".json", f"_{split}_episodes.json")
            episodes = []
            for r in results:
                ep = {"game_file": r["game_file"], "success": r["success"], "steps": r["steps"]}
                if r.get("error"):
                    ep["error"] = r["error"]
                if not r["success"] and r.get("trajectory"):
                    ep["trajectory"] = r["trajectory"]
                episodes.append(ep)
            os.makedirs(os.path.dirname(traj_file), exist_ok=True)
            with open(traj_file, "w") as f:
                json.dump(episodes, f, ensure_ascii=False)
            logger.info(f"  Episode details saved to {traj_file}")

    logger.info("=" * 50)
    logger.info("FINAL RESULTS:")
    for split, stats in all_results.items():
        logger.info(f"  {split}: SR = {stats['sr']:.4f} ({stats['successes']}/{stats['valid']})")

    # Save results
    output_file = args.output or f"/tmp/areal/eval_results_{int(time.time())}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    await env.aclose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", default="/storage/openpsi/users/yl/agent-memory/MemRL/data/alfworld/json_2.1.1")
    parser.add_argument("--splits", nargs="+", default=["train", "valid_seen", "valid_unseen"])
    parser.add_argument("--max_episodes", type=int, default=200, help="Max episodes per split")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--save_trajectories", action="store_true", help="Save failure trajectories for analysis")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--env_port", type=int, default=8765)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
