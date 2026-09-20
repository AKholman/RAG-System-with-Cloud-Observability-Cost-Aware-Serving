from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import json
import time
import uuid

from app.logger import logger
from app.s3_loader import download
from app.retrieval import load, search
from app.groq_client import generate


cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    download()
    load()
    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    k: int = 10

@app.post("/query")
def query_endpoint(req: QueryRequest):

    request_id = str(uuid.uuid4())
    total_start = time.time()

    try:
        logger.info(json.dumps({
            "event": "query_received",
            "request_id": request_id,
            "query": req.query
        }))

        # 1. Retrieval
        retrieval_start = time.time()

        documents = search(req.query, k=req.k)

        retrieval_ms = (time.time() - retrieval_start) * 1000

        # Extract document IDs
        retrieved_doc_ids = [
            doc.get("doc_id") for doc in documents
        ]

        # Build context
        context = "\n\n".join(
            doc.get("text", "") for doc in documents
        )

        # 2. Generation
        answer, token_usage, llm_ms = generate(
            req.query,
            context
        )

        # 3. Total latency
        total_latency_ms = (time.time() - total_start) * 1000

        # 4. CloudWatch
        cloudwatch.put_metric_data(
            Namespace="RAGService",
            MetricData=[
                {
                    "MetricName": "RetrievalLatencyMs",
                    "Value": retrieval_ms,
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

        logger.info(json.dumps({
            "event": "query_completed",
            "request_id": request_id,
            "query": req.query,
            "retrieved_docs": retrieved_doc_ids,
            "retrieval_latency_ms": retrieval_ms,
            "llm_latency_ms": llm_ms,
            "total_latency_ms": total_latency_ms,
            "token_usage": token_usage
        }))

        return {
            "request_id": request_id,
            "query": req.query,
            "retrieved_docs": retrieved_doc_ids,
            "context": context,
            "answer": answer,
            "latency_ms": {
                "retrieval": retrieval_ms,
                "llm": llm_ms,
                "total": total_latency_ms
            },
            "token_usage": token_usage
        }


    except Exception as e:

        logger.error(json.dumps({
            "event": "query_failed",
            "request_id": request_id,
            "query": req.query,
            "error": str(e)
        }))

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
