import os
import time
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def generate(query: str, context: str) -> tuple:
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer based only on the provided context."
    )

    start = time.time()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful RAG assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512
    )

    latency_ms = (time.time() - start) * 1000

    usage = response.usage

    token_usage = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }

    return response.choices[0].message.content, token_usage, latency_ms
