import time

import openai
from openai.types.chat import ChatCompletion


def make_request(
    client: openai.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    reasoning_effort: str = "medium",
    n: int = 1,
    **kwargs
) -> ChatCompletion:
    kwargs["top_p"] = 0.95
    kwargs["max_completion_tokens"] = max_tokens
    kwargs["temperature"] = temperature
    if any(model.startswith(m) or model.endswith(m) for m in ["o1-", "o3-", "reasoner", "grok-3-mini-beta"]):  # pop top-p and max_completion_tokens
        kwargs.pop("top_p")
        kwargs.pop("max_completion_tokens")
        kwargs.pop("temperature")
        kwargs["reasoning_effort"] = reasoning_effort
    
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": message},
        ],
        n=n,
        **kwargs
    )


def make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret