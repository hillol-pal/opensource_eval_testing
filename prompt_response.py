
import os
from dotenv import load_dotenv
from openai import OpenAI

# load environment variables from .env file
load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(prompt:str) -> str:
    """
    Generates a response to the given prompt using an AI model.

    Args:
        prompt (str): The input prompt for which a response is to be generated.

    Returns:
        str: The generated response from the AI model.
    """
    # Placeholder for actual AI model call
    # In practice, this would involve calling an API or a local model
    client_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    response = client_response.choices[0].message.content
    return response




