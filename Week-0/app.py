from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from your .env file

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{
        "role": "user",
        "content": "What is Generative AI in exactly 2 sentences?"
    }]
)

print(response.choices[0].message.content)