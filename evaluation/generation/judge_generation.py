import json
import os
import re
from groq import Groq

INPUT_FILE = "evaluation/data/generated_answers.json"
OUTPUT_FILE = "evaluation/reports/generation_metrics.json"

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "qwen/qwen3.8-27b"


def judge(item):
    prompt = f"""
Evaluate this RAG answer.

Question:
{item["query"]}

Reference answer:
{item["reference_answer"]}

Retrieved context:
{item["context"]}

Generated answer:
{item["generated_answer"]}

Score each from 0 to 2:

Correctness:
0 incorrect, 1 partially correct, 2 correct

Faithfulness:
0 unsupported, 1 partially supported, 2 fully supported

Relevance:
0 does not answer, 1 partially answers, 2 directly answers

Return ONLY JSON:
{{
  "correctness": 0,
  "faithfulness": 0,
  "relevance": 0
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100 
    )

    text = response.choices[0].message.content
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"Invalid judge response: {text}")

    return json.loads(match.group())


def main():
    with open(INPUT_FILE) as f:
        items = json.load(f)

    results = []

    for i, item in enumerate(items, 1):
        print(f"Judging {i}/{len(items)}")
        scores = judge(item)

        results.append({
            "query": item["query"],
            "generated_answer": item["generated_answer"],
            "scores": scores
        })

    n = len(results)

    report = {
        "num_queries": n,
        "average_correctness": sum(
            x["scores"]["correctness"] for x in results
        ) / n,
        "average_faithfulness": sum(
            x["scores"]["faithfulness"] for x in results
        ) / n,
        "average_relevance": sum(
            x["scores"]["relevance"] for x in results
        ) / n,
        "results": results
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("\nGeneration evaluation complete.")
    print(json.dumps({
        k: report[k]
        for k in [
            "num_queries",
            "average_correctness",
            "average_faithfulness",
            "average_relevance"
        ]
    }, indent=2))


if __name__ == "__main__":
    main()
