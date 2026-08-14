import time

from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types import TextGenerationOutput


def make_request(
    client: InferenceClient,
    message: str,
    model: str,
    temperature: float,
    n: int,
    max_new_tokens: int = 2048,
) -> TextGenerationOutput:
    response = client.text_generation(
        model=model,
        prompt=message,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    return response


def make_auto_request(*args, **kwargs) -> TextGenerationOutput:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret
