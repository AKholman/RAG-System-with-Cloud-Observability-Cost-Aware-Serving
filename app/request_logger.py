import time
import uuid
from app.logger import logger

def log_request(query, top_k_ids, embed_time, faiss_time, llm_time, tokens=None, error=None):

    log_data = {
        "request_id": str(uuid.uuid4()),
        "query": query,
        "top_k_docs": top_k_ids,
        "latency_ms": {
            "embedding": embed_time,
            "faiss": faiss_time,
            "llm": llm_time
        },
        "tokens": tokens,
        "error": str(error) if error else None
    }

    logger.info(log_data)
