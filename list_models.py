from os import getenv

from google import genai


client = genai.Client(api_key=getenv("GEMINI_API_KEY"))
for model in client.models.list():
    print(model.name)
