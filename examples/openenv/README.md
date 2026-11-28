# OpenEnv rollouts

This example connects to an existing OpenEnv Echo server. The environment backend opens
one client per rollout and closes the connection when the rollout exits.

Install the Echo client from an OpenEnv checkout in the same Python environment as AReaL
with `python -m pip install /path/to/openenv /path/to/openenv/envs/echo_env`. Follow the
[Echo environment instructions](https://github.com/huggingface/openenv/blob/main/docs/source/environments/echo.md)
to start a server and set `OPENENV_BASE_URL` to its URL. To try the adapter without
model inference, run from the AReaL root.

```bash
python -m examples.openenv.agent --env-url "$OPENENV_BASE_URL" --message 'hello'
```

To use the agent with `OpenAIProxyWorkflow`, set `examples.openenv.agent.EchoAgent` as
the workflow in an inline agent run, supply dataset rows with a `messages` list, and
pass `OPENENV_BASE_URL` to the rollout workers. The agent uses the proxy's per-session
credentials and turns each text completion into an `echo_message` action. Echo supplies
no learning reward, so this wiring demo assigns zero when it reaches the turn limit.

For `reward_mode="dense"`, set `agent.export_style` to `individual` in the inference
engine config. This exports each completion with its reward after turn discounting.
`missing_reward_policy` can reject the rollout, omit missing assignments, set them to
zero, or fill them from the last step. If every reward is missing, `ignore` passes an
empty map to the proxy. Individual export treats omitted assignments as zero before
propagating later rewards.

Use `truncation_policy` to drop truncated episodes, retain them with their last step's
reward, or assign zero to that step. Environment termination is exported as interaction
metadata. PPO's `is_truncated` continues to describe the model's generation length
limit.
