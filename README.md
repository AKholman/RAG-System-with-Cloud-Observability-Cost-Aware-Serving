
# 🧠 RAG System with Cloud Observability & Cost-Aware LLM Serving

A **production-style Retrieval-Augmented Generation (RAG) system** built with a clear separation between **offline data processing** and **online inference**, designed to run on **free / low-cost cloud resources**.

This project demonstrates how to build, deploy, evaluate, and monitor a real RAG system end-to-end.

---

## 🔍 Project does

* Turns large text data into **semantic search indexes**
* Serves **low-latency RAG queries** via an API
* Uses **FAISS-only retrieval** 
* Generates answers with **Groq LLaMA-3.1-8B-Instant**
* Tracks **latency, token usage, and logs** in CloudWatch

---

## 🧱 End-to-End Workflow

### 1️⃣ Offline Pipeline (Google Colab – GPU)

Used only for heavy batch processing.

**Steps**

1. Download real-world news data (CC News)
2. Clean and normalize text
3. Chunk documents into ~300–400 token segments
4. Generate embeddings using SentenceTransformer `all-MiniLM-L6-v2` (GPU)
5. Build FAISS vector index (cosine similarity)
6. Save metadata (chunk → document mapping)
7. Upload artifacts to **AWS S3**

**Output artifacts**

* `faiss.index`
* `metadata.json`

These artifacts are reused by the online system (no recomputation).

---

### 2️⃣ Online API (AWS EC2 – CPU only)

Runs on a **t3.micro** instance.

## ▶️ How to Run (EC2)

1. Upload FAISS artifacts to S3
2. Launch EC2 (t3.micro)
3. Install dependencies
4. Start API:
   uvicorn app.main:app --host 0.0.0.0 --port 8000
5. Query:
   POST /query

**Request flow**

```
User Query
 → SentenceTransformer embedding
 → FAISS Top-K semantic search
 → Context assembly
 → Groq LLaMA-3.1-8B-Instant
 → Final answer
```

**Key points**

* No GPU on EC2
* FAISS loaded once into RAM
* LLM inference handled by Groq API
* FastAPI used for serving

---

## 📊 Evaluation Strategy

### Retrieval

* Recall@K
* MRR@K

### Generation (RAG vs LLM-only)

* Compare:

  * LLaMA + FAISS context (RAG)
  * LLaMA without retrieval
* Focus on grounding and factual consistency

**Retrieval results (50 queries)**
    
    "num_queries": 50,
    "Recall@5": 0.86,
    "Recall@10": 0.88,
    "MRR@10": 0.6982
     

---

## 📈 Monitoring & Observability (AWS CloudWatch)

Production-style monitoring from EC2 logs.

### Metrics (1-minute resolution)

| Metric            | Avg         |
| ----------------- | ----------- |
| Embedding Latency | ~50 ms      |
| FAISS Latency     | ~20 ms      |
| LLM Latency       | ~200 ms     |
| **Total Latency** | **~270 ms** |

**Token usage**

* Avg tokens/query ≈ 130
* Total ≈ 3.77k
* Cost < $0.01 (Groq)

**Logs**

* 60 structured request traces
* p95 latency ≈ 270 ms
* 0 errors

---

## 🗂️ Repository Structure

```
rag-api/
├── app/                       # Online inference service (EC2)
│   ├── main.py                # FastAPI entrypoint
│   ├── retrieval.py           # FAISS retrieval + query pipeline
│   ├── s3_loader.py           # Load FAISS + metadata from S3
│   ├── groq_client.py         # Groq LLaMA API wrapper
│   ├── logger.py              # Structured logging
│   └── request_logger.py      # Per-request tracing & latency metrics
│
├── evaluation/                # Offline evaluation & analysis
│   ├── run_eval.py            # Main evaluation entrypoint
│   │
│   ├── retrieval/             # Retrieval-quality evaluation
│   │   └── eval_retrieval.py  # Recall@K, MRR
│   │
│   ├── generation/            # Generation-quality evaluation
│   │   └── eval_generation.py # RAG vs LLM-only comparison
│   │
│   ├── data/                  # Evaluation datasets
│   │   ├── eval_queries.json
│   │   └── gold_queries_balanced.json
│   │
│   └── reports/               # Evaluation outputs
│       ├── retrieval_metrics.json
│       └── results.json
│
└── README.md
```
---

## 🧠 Design Decisions

- Offline GPU processing (Colab) to avoid costly always-on GPUs
- CPU-only EC2 serving for stability and cost control
- FAISS in-memory index for low-latency retrieval
- External LLM API (Groq) to decouple inference from infra
- CloudWatch logging to enable production-style observability

---

## 🚀 Future Improvements

* Cross-encoder reranker
* Caching layer (Redis)
* Streaming responses
* Docker + Load Balancer
* Managed vector database

---

