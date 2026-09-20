import json
import os
import requests
import numpy as np

API_URL = "http://localhost:8000/query"
QUERIES_FILE = "evaluation/data/eval_queries.json"
DOCUMENTS_FILE = "evaluation/data/eval_documents.json"
OUTPUT_FILE = "evaluation/reports/retrieval_metrics.json"


def normalize_id(value):
    return str(value)


def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved = {
        normalize_id(x)
        for x in retrieved_ids[:k]
    }

    relevant = {
        normalize_id(x)
        for x in relevant_ids
    }

    return 1.0 if retrieved.intersection(relevant) else 0.0


def mrr_at_k(retrieved_ids, relevant_ids, k):
    relevant = {
        normalize_id(x)
        for x in relevant_ids
    }

    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if normalize_id(doc_id) in relevant:
            return 1.0 / rank

    return 0.0


def main():

    with open(QUERIES_FILE) as f:
        queries = json.load(f)

    with open(DOCUMENTS_FILE) as f:
        documents = json.load(f)

    document_ids = {
        normalize_id(doc["doc_id"])
        for doc in documents
    }

    for item in queries:
        for doc_id in item["relevant_doc_ids"]:
            if normalize_id(doc_id) not in document_ids:
                raise ValueError(
                    f"Invalid relevant_doc_id: {doc_id}"
                )

    recall5 = []
    recall10 = []
    mrr5 = []
    mrr10 = []
    per_query = []

    print(f"Queries: {len(queries)}")
    print(f"Evaluation documents: {len(documents)}")
    print()

    for i, item in enumerate(queries, start=1):

        query = item["query"]
        relevant_ids = item["relevant_doc_ids"]

        print(f"[{i}/{len(queries)}] {query}")

        response = requests.post(
            API_URL,
            json={"query": query, "k": 10},
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        retrieved_ids = data.get("retrieved_docs", [])

        r5 = recall_at_k(
            retrieved_ids,
            relevant_ids,
            5
        )

        r10 = recall_at_k(
            retrieved_ids,
            relevant_ids,
            10
        )

        m5 = mrr_at_k(
            retrieved_ids,
            relevant_ids,
            5
        )

        m10 = mrr_at_k(
            retrieved_ids,
            relevant_ids,
            10
        )

        rank = None

        relevant_set = {
            normalize_id(x)
            for x in relevant_ids
        }

        for position, doc_id in enumerate(
            retrieved_ids,
            start=1
        ):
            if normalize_id(doc_id) in relevant_set:
                rank = position
                break

        recall5.append(r5)
        recall10.append(r10)
        mrr5.append(m5)
        mrr10.append(m10)

        per_query.append({
            "query": query,
            "relevant_doc_ids": relevant_ids,
            "retrieved_doc_ids": retrieved_ids,
            "relevant_rank": rank,
            "recall@5": r5,
            "recall@10": r10,
            "mrr@5": m5,
            "mrr@10": m10
        })

    results = {
        "metrics": {
            "num_queries": len(queries),
            "recall@5": float(np.mean(recall5)),
            "recall@10": float(np.mean(recall10)),
            "mrr@5": float(np.mean(mrr5)),
            "mrr@10": float(np.mean(mrr10))
        },
        "per_query": per_query
    }

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\nRetrieval evaluation complete.")
    print(json.dumps(results["metrics"], indent=2))
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
