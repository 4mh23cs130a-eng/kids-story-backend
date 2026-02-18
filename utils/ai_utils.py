import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

endpoint = "https://models.github.ai/inference"
model = "gpt-4o-mini"
token = os.environ["GITHUB_TOKEN"]
print(token)
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

def generate_story(prompt: str):
    try:
        response = client.complete(
            messages=[
                SystemMessage("You are a professional children's book author. Write engaging, simple, and moral stories suitable for kids. Keep the language simple and the tone positive."),
                UserMessage(f"Write a short story based on this prompt: {prompt}"),
            ],
            model=model
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating story: {e}")
        return "I'm sorry, I couldn't generate a story right now. Please try again later."
