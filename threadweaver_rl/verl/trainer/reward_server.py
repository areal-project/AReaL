import os
import traceback
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request
from transformers import AutoTokenizer
from deepscaler.rewards.math_rewardv2 import deepscaler_reward_fn

# --- Reward Function Selection ---
def _select_rm_score_fn(data_source: str):
    """Selects the reward function based on the data source."""
    # This can be extended to route to different reward models.
    if "math" in data_source:
        return deepscaler_reward_fn
    return deepscaler_reward_fn

# --- Flask App Factory ---
def create_app():
    """Creates and configures a Flask app for the reward server."""
    app = Flask(__name__)

    # Disable tokenizer parallelism to prevent issues when used with Gunicorn workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load tokenizer once at startup using the environment variable
    tokenizer_path = os.environ.get("TOKENIZER_PATH")
    if not tokenizer_path:
        raise RuntimeError("TOKENIZER_PATH environment variable not set.")
    app.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"✅ Reward server tokenizer loaded from: {tokenizer_path}")

    @app.route("/compute_reward", methods=["POST"])
    def compute_reward():
        """
        Main endpoint for computing rewards.
        It handles requests containing either 'sequence_str' or 'sequence_ids'.
        """
        payload = request.get_json()
        if not payload:
            error_msg = "Invalid request: no JSON payload found."
            print(error_msg)

            return jsonify({"error": error_msg}), 400

        # --- Extract and Validate Input ---
        sequence_str = payload.get("sequence_str")
        sequence_ids = payload.get("sequence_ids")

        if sequence_str is None and sequence_ids is None:
            error_msg = "Either 'sequence_str' or 'sequence_ids' must be provided."
            print(error_msg)

            return jsonify({"error": error_msg}), 400

        # If IDs are provided, decode them. 'sequence_str' takes precedence.
        if sequence_str is None:
            try:
                sequence_str = app.tokenizer.decode(sequence_ids)
            except Exception as e:
                error_msg = f"Failed to decode sequence_ids: {e}"
                print(error_msg)

                return jsonify({"error": error_msg}), 400

        ground_truth = payload.get("ground_truth")
        data_source = payload.get("data_source", "unknown")
        correctness_as_reward = payload.get("correctness_as_reward", False)
        skip_reward_fn = payload.get("skip_reward_fn", False)
        config = payload.get("config", {})

        # --- Compute Score ---
        compute_score_fn = _select_rm_score_fn(data_source)
        try:
            score, extra_info = compute_score_fn(
                solution_str=sequence_str,
                ground_truth=ground_truth,
                config=config,
                correctness_as_reward=correctness_as_reward,
                skip_reward_fn=skip_reward_fn,
                tokenizer=app.tokenizer,
            )
        except Exception as e:
            error_msg = f"Error during reward computation: {e}"
            print(error_msg)

            # Handle potential errors within the reward function itself
            return jsonify({"error": error_msg}), 500

        # --- Return Computed Result ---
        return jsonify({"score": score, "extra_info": extra_info})

    return app
