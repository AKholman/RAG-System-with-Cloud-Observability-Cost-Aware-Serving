import json
import os
import requests

API_URL = "http://localhost:8000/query"
INPUT_FILE = "evaluation/data/eval_queries.json"
OUTPUT_FILE = "evaluation/data/generated_answers.json"


def main():
    with open(INPUT_FILE) as f:
        queries = json.load(f)

    results = []

    for i, item in enumerate(queries, 1):
        print(f"Generating {i}/{len(queries)}: {item['query']}")

        response = requests.post(
            API_URL,
            json={"query": item["query"], "k": 10},
            timeout=120
        )
        response.raise_for_status()

        data = response.json()

        results.append({
            "query": item["query"],
            "reference_answer": item["reference_answer"],
            "retrieved_docs": data["retrieved_docs"],
            "context": data["context"],
            "generated_answer": data["answer"]
        })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Generated answers: {len(results)}")


if __name__ == "__main__":
    main()
