import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer, util
from app.retrieval import search
from app.groq_client import generate

embedder = SentenceTransformer("all-MiniLM-L6-v2")

with open("evaluation/eval_queries.json") as f:
    queries = json.load(f)

results = []

for q in queries:
    query = q["query"]
    print(f"Evaluating: {query}")

    # RAG retrieval
    docs = search(query, k=5)
    context = "\n".join([d.get("text", "") for d in docs])

    # RAG answer (with context)
    rag_answer = generate(query, context=context)

    # LLM-only answer (no context)
    base_answer = generate(query, context=None)

    # semantic similarity
    e1 = embedder.encode(rag_answer, convert_to_tensor=True)
    e2 = embedder.encode(base_answer, convert_to_tensor=True)
    similarity = float(util.cos_sim(e1, e2))

    print(f"  similarity: {similarity:.3f}")

    results.append({
        "query": query,
        "retrieved_context": context,
        "rag_answer": rag_answer,
        "llm_only_answer": base_answer,
        "semantic_similarity": similarity
    })

with open("evaluation/generation_eval.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Results saved to evaluation/generation_eval.json")
