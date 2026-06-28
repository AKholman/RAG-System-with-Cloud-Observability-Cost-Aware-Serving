from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate(query: str, context: str = None) -> str:
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based only on the context."
    else:
        prompt = f"Question: {query}\nAnswer based on your knowledge."

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512
    )
    return response.choices[0].message.content
