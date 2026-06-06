from openai import OpenAI

# Modify to your server's address
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-if-needed")

response = client.completions.create(model="/storage/openpsi/models/Qwen__Qwen3-4B", prompt="Hello!")
print(response)