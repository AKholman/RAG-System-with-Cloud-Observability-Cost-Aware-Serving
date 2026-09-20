import json
import numpy as np
import os

LOG_FILE = "logs/app.log"
QUERY_FILE = "evaluation/data/eval_queries.json"
OUTPUT_FILE = "evaluation/reports/performance_metrics.json"


def percentile(values, p):
    return round(float(np.percentile(values, p)), 2)


# Load evaluation queries
with open(QUERY_FILE) as f:
    eval_queries = json.load(f)

# We know the latest run completed the first 36 queries
target_queries = [item["query"] for item in eval_queries[:36]]

# Collect all completed log records
completed = []

with open(LOG_FILE) as f:
    for line in f:
        try:
            record = json.loads(line)
            message = record.get("message", {})

            if message.get("event") == "query_completed":
                completed.append(message)

        except (json.JSONDecodeError, KeyError):
            continue


# Find the latest occurrence of each of the 36 queries
latest = {}

for message in completed:
    query = message.get("query")

    if query in target_queries:
        latest[query] = message


# Preserve evaluation order
records = [
    latest[q]
    for q in target_queries
    if q in latest
]

retrieval = [r["retrieval_latency_ms"] for r in records]
llm = [r["llm_latency_ms"] for r in records]
total = [r["total_latency_ms"] for r in records]


report = {
    "num_completed_queries": len(records),
    "retrieval_latency_ms": {
        "p50": percentile(retrieval, 50),
        "p95": percentile(retrieval, 95)
    },
    "llm_latency_ms": {
        "p50": percentile(llm, 50),
        "p95": percentile(llm, 95)
    },
    "total_latency_ms": {
        "p50": percentile(total, 50),
        "p95": percentile(total, 95)
    }
}


os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
