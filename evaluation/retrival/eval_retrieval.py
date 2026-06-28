import json
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval import search

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gold", required=True)
args = parser.parse_args()

with open(args.gold) as f:
    gold = json.load(f)

recall_5 = []
recall_10 = []
mrrs = []

for item in gold:
    query = item["query"]
    relevant = set(item["relevant_doc_ids"])

    # Retrieve top 10
    results_10 = search(query, k=10)
    retrieved_ids_10 = [r["doc_id"] for r in results_10]
    retrieved_ids_5 = retrieved_ids_10[:5]

    # Recall@5
    hits_5 = [i for i, doc_id in enumerate(retrieved_ids_5) if doc_id in relevant]
    recall_5.append(len(hits_5) > 0)

    # Recall@10
    hits_10 = [i for i, doc_id in enumerate(retrieved_ids_10) if doc_id in relevant]
    recall_10.append(len(hits_10) > 0)

    # MRR@10
    first_hit = next((i + 1 for i, doc_id in enumerate(retrieved_ids_10) if doc_id in relevant), None)
    mrrs.append(1 / first_hit if first_hit else 0)

metrics = {
    "num_queries": len(gold),
    "Recall@5":  round(float(np.mean(recall_5)), 4),
    "Recall@10": round(float(np.mean(recall_10)), 4),
    "MRR@10":    round(float(np.mean(mrrs)), 4)
}

with open("evaluation/retrieval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== Retrieval Metrics ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print("\nSaved to evaluation/retrieval_metrics.json")
