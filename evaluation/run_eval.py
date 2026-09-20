import json

with open("evaluation/reports/retrieval_metrics.json") as f:
    r = json.load(f)["metrics"]

with open("evaluation/reports/generation_metrics.json") as f:
    g = json.load(f)

with open("evaluation/reports/performance_metrics.json") as f:
    p = json.load(f)

print("\nRAG Evaluation Summary")
print("=" * 30)

print("\nRetrieval:")
print(f"  Recall@5:  {r['recall@5']}")
print(f"  Recall@10: {r['recall@10']}")
print(f"  MRR@5:     {r['mrr@5']}")
print(f"  MRR@10:    {r['mrr@10']}")

print("\nGeneration:")
print(f"  Correctness:  {g['average_correctness']}")
print(f"  Faithfulness: {g['average_faithfulness']}")
print(f"  Relevance:    {g['average_relevance']}")

print("\nPerformance:")
print(f"  Queries:       {p['num_completed_queries']}")
print(f"  Retrieval P50: {p['retrieval_latency_ms']['p50']} ms")
print(f"  Retrieval P95: {p['retrieval_latency_ms']['p95']} ms")
print(f"  LLM P50:       {p['llm_latency_ms']['p50']} ms")
print(f"  LLM P95:       {p['llm_latency_ms']['p95']} ms")
print(f"  Total P50:     {p['total_latency_ms']['p50']} ms")
print(f"  Total P95:     {p['total_latency_ms']['p95']} ms")
