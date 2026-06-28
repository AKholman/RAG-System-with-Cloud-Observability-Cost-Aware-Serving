from fastapi import FastAPI
from pydantic import BaseModel
import uuid, time, json, boto3

from app.logger import logger

app = FastAPI()

cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")


class QueryRequest(BaseModel):
    query: str


def call_groq(query: str):
    start = time.time()

    time.sleep(0.2)  # simulate Groq latency

    answer = f"Mock response for: {query}"

    token_usage = {
        "prompt_tokens": 50,
        "completion_tokens": 80,
        "total_tokens": 130
    }

    latency_ms = (time.time() - start) * 1000

    return answer, token_usage, latency_ms


@app.post("/query")
def query_endpoint(req: QueryRequest):

    request_id = str(uuid.uuid4())

    try:
        logger.info(json.dumps({
            "event": "query_received",
            "request_id": request_id,
            "query": req.query
        }))

        # =========================
        # 1. EMBEDDING (mock)
        # =========================
        t0 = time.time()
        time.sleep(0.05)
        embedding_ms = (time.time() - t0) * 1000

        # =========================
        # 2. FAISS SEARCH (mock)
        # =========================
        t1 = time.time()
        time.sleep(0.02)
        faiss_ms = (time.time() - t1) * 1000

        # IMPORTANT: needed for evaluation
        retrieved_doc_ids = [1, 2, 3]

        # =========================
        # 3. LLM (Groq)
        # =========================
        answer, token_usage, llm_ms = call_groq(req.query)

        # =========================
        # 4. TOTAL LATENCY
        # =========================
        total_latency_ms = embedding_ms + faiss_ms + llm_ms

        # =========================
        # 5. CLOUDWATCH METRICS
        # =========================
        cloudwatch.put_metric_data(
            Namespace="RAGService",
            MetricData=[
                {
                    "MetricName": "EmbeddingLatencyMs",
                    "Value": embedding_ms,
                    "Unit": "Milliseconds"
                },
                {
                    "MetricName": "FAISSSearchLatencyMs",
                    "Value": faiss_ms,
                    "Unit": "Milliseconds"
                },
                {
                    "MetricName": "LLMLatencyMs",
                    "Value": llm_ms,
                    "Unit": "Milliseconds"
                },
                {
                    "MetricName": "TotalLatencyMs",
                    "Value": total_latency_ms,
                    "Unit": "Milliseconds"
                },
                {
                    "MetricName": "TotalTokens",
                    "Value": token_usage["total_tokens"],
                    "Unit": "Count"
                }
            ]
        )

        # =========================
        # SUCCESS LOG
        # =========================
        logger.info(json.dumps({
            "event": "query_completed",
            "request_id": request_id,
            "query": req.query,
            "retrieved_docs": retrieved_doc_ids,   # ✅ REQUIRED FOR EVAL
            "retrieval": {
                "top_k_docs": retrieved_doc_ids,
                "embedding_latency_ms": embedding_ms,
                "faiss_latency_ms": faiss_ms
            },
            "generation": {
                "llm_latency_ms": llm_ms,
                "token_usage": token_usage
            },
            "total_latency_ms": total_latency_ms
        }))

        return {
            "request_id": request_id,
            "query": req.query,
            "retrieved_docs": retrieved_doc_ids,   # ✅ ADDED FOR EVAL
            "answer": answer
        }

    except Exception as e:
        logger.error(json.dumps({
            "event": "query_failed",
            "request_id": request_id,
            "query": req.query,
            "error": str(e)
        }))
        return {"error": "internal error"}
