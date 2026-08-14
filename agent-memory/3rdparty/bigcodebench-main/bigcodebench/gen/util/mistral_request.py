import time

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

def make_auto_request(client: MistralClient, *args, **kwargs) -> ChatMessage:
    ret = None
    while ret is None:
        try:
            ret = client.chat(*args, **kwargs)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret