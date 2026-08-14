import time

from google import genai
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted


def make_request(
    model: str,
    client: genai.Client,
    message: str,
    temperature: float,
    n: int,
    max_new_tokens: int = 2048,
) -> genai.types.GenerateContentResponse:
    kwargs = {"temperature": temperature, "max_output_tokens": max_new_tokens}

    if "-thinking-" in model:
        kwargs.pop("max_output_tokens")
    
    response = client.models.generate_content(
        model=model,
        contents=message,
        config=genai.types.GenerateContentConfig(
            candidate_count=n,
            safety_settings=[
                genai.types.SafetySetting(
                    category='HARM_CATEGORY_DANGEROUS_CONTENT',
                    threshold='BLOCK_NONE'
                ),
                genai.types.SafetySetting(
                    category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    threshold='BLOCK_NONE'
                ),
                genai.types.SafetySetting(
                    category='HARM_CATEGORY_HATE_SPEECH',
                    threshold='BLOCK_NONE'
                ),
                genai.types.SafetySetting(
                    category='HARM_CATEGORY_HARASSMENT',
                    threshold='BLOCK_NONE'
                ),
            ],
            **kwargs
        ),            
    )

    return response


def make_auto_request(*args, **kwargs) -> genai.types.GenerateContentResponse:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except ResourceExhausted as e:
            print("Rate limit exceeded. Waiting...", e.message)
            time.sleep(10)
        except GoogleAPICallError as e:
            print(e.message)
            time.sleep(1)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret