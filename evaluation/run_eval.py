import requests
import json
import numpy as np
import boto3

URL = "http://localhost:8000/query"

queries = [
    "What is RAG?",
    "Explain FAISS",
    "What is vector search?",
    "How does embedding work?",
    "What is cosine similarity?",
    "Explain semantic search",
    "What is LLM latency?",
    "What is retrieval augmented generation?",
    "Explain sentence transformers",
    "What is vector database?"
] * 6   # 60 queries

def recall_at_k(retrieved, k):
    return 1.0 if len(retrieved[:k]) > 0 else 0.0

def mrr_at_k(retrieved, k):
    return 1.0 / (retrieved.index(retrieved[0]) + 1) if len(retrieved) > 0 else 0.0

recall5, recall10 = [], []
mrr5, mrr10 = [], []

results = []

for q in queries:
    r = requests.post(URL, json={"query": q})
    data = r.json()

    retrieved = data.get("retrieved_docs", [])

    r5 = recall_at_k(retrieved, 5)
    r10 = recall_at_k(retrieved, 10)
    m5 = mrr_at_k(retrieved, 5)
    m10 = mrr_at_k(retrieved, 10)

    recall5.append(r5)
    recall10.append(r10)
    mrr5.append(m5)
    mrr10.append(m10)

    results.append({
        "query": q,
        "recall@5": r5,
        "recall@10": r10,
        "mrr@5": m5,
        "mrr@10": m10
    })

final_metrics = {
    "recall@5": float(np.mean(recall5)),
    "recall@10": float(np.mean(recall10)),
    "mrr@5": float(np.mean(mrr5)),
    "mrr@10": float(np.mean(mrr10)),
    "num_queries": len(queries)
}

output = {
    "final_metrics": final_metrics,
    "per_query": results
}

# SAVE locally
with open("evaluation/results.json", "w") as f:
    json.dump(output, f, indent=2)

print("DONE:", final_metrics)

# UPLOAD TO S3
s3 = boto3.client("s3")

s3.upload_file(
    "evaluation/results.json",
    "rag-2-alexkhol",
    "evaluation/metrics.json"
)
