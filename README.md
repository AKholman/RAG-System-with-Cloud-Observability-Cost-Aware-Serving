
````markdown
# RAG System with AWS Deployment & Cloud Observability

Production-oriented Retrieval-Augmented Generation (RAG) system deployed on AWS EC2 with S3 artifact storage, FAISS semantic retrieval, Groq-hosted `openai/gpt-oss-20b` inference, and CloudWatch monitoring.

The system separates **offline batch processing** from **online inference**, allowing computationally expensive embedding and indexing operations to be performed once and reused by the production API.

## Key Capabilities

- Semantic document retrieval using **Sentence Transformers + FAISS**
- **AWS S3** storage for reusable retrieval artifacts
- **FastAPI** inference service running on an AWS EC2 `t3.micro`
- LLM generation using **Groq `openai/gpt-oss-20b`**
- Request-level logging with latency and token usage
- **Amazon CloudWatch** custom metrics for production observability
- Separate evaluation of retrieval quality, generation quality, and inference performance

---

## Architecture

```text
                    OFFLINE LAYER
                 Colab GPU / Batch Job
                         │
                  CC News dataset
                         │
                  Text cleaning
                         │
                  Sentence-aware
                     chunking
                         │
              all-MiniLM-L6-v2
                384-d embeddings
                         │
                 Normalize vectors
                         │
              FAISS IndexFlatIP
                         │
             ┌───────────┴───────────┐
             │                       │
        faiss.index             metadata.json
             │                       │
             └───────────┬───────────┘
                         │
                    Amazon S3
                         │
                         ▼
                 ONLINE INFERENCE
                  AWS EC2 t3.micro
                         │
                    FastAPI /query
                         │
                  Query embedding
                         │
                  FAISS Top-K search
                         │
                 Retrieved documents
                         │
                  Context assembly
                         │
                         ▼
             Groq openai/gpt-oss-20b
                         │
                         ▼
                    Final answer
                         │
              ┌──────────┴──────────┐
              │                     │
         Local logging         CloudWatch
         app.log               metrics
````
====================================
### Offline Processing layer
====================================
The offline layer processes approximately **100K CC News articles (~393 MB)**:

1. Clean article text.
2. Split text at sentence boundaries.
3. Create chunks of up to 400 whitespace-separated words.
4. Generate 384-dimensional embeddings using `all-MiniLM-L6-v2`.
5. Normalize embeddings for cosine-similarity search.
6. Build a FAISS `IndexFlatIP` exact nearest-neighbor index.
7. Save the index and document metadata.
8. Upload artifacts to Amazon S3.

The resulting artifacts are reused during inference; embeddings and indexing are **not recomputed for each query**.

======================================
### Online Inference layer
======================================
The EC2 service performs:

```text
User Query
    ↓
SentenceTransformer embedding
    ↓
FAISS Top-K semantic retrieval
    ↓
Context assembly
    ↓
Groq openai/gpt-oss-20b
    ↓
Final Answer
```

The API loads `faiss.index` and `metadata.json` from S3 when the service starts.


### AWS Components

* **EC2 t3.micro** — FastAPI inference service
* **S3** — FAISS index and metadata storage
* **CloudWatch** — application performance metrics

==================================================================
### Application Logging & Cloudwatch - monitoring and observation
==================================================================

`app.log` records:

* Request ID
* Query
* Retrieved documents
* Retrieval latency
* LLM latency
* Total request latency
* Token usage

CloudWatch custom metrics track retrieval latency, LLM latency, total latency, and token usage.

==========================================
## Evaluation  (offline)
==========================================
The system was evaluated across **retrieval quality, generation quality, and production latency**.

### Retrieval
50-query benchmark derived from the evaluation corpus:

| Metric    | Result |
| --------- | -----: |
| Recall@5  |   0.68 |
| Recall@10 |   0.70 |
| MRR@5     |  0.509 |
| MRR@10    |  0.512 |

### Generation

The RAG answers were evaluated by a separate LLM judge using `qwen/qwen3.8-27b`.

Three-level scoring:
* **0** — incorrect / unsupported / irrelevant
* **1** — partially correct / supported / relevant
* **2** — fully correct / supported / relevant

| Metric       |   Result |
| ------------ | -------: |
| Correctness  | 1.38 / 2 |
| Faithfulness | 1.46 / 2 |         # To estimate hallucination
| Relevance    | 1.50 / 2 |

### Performance  (36 completed queries.)

Measured on the deployed EC2 inference service:

| Metric             |     P50 |      P95 |
| ------------------ | ------: | -------: |
| Retrieval latency  | 86.8 ms | 223.5 ms |
| LLM latency        |  18.4 s |   21.3 s |
| End-to-end latency |  18.5 s |   21.3 s |

-----
### Evaluation Limitations

The retrieval and generation benchmarks contain 50 questions mapped to known relevant documents in the evaluation corpus. This provides a reproducible benchmark but does not fully represent open-ended production traffic, such as differently phrased questions, multi-document questions, or questions with no answer in the corpus.

Performance testing included 36 completed queries because the external LLM API quota was reached during the 50-query test.

---
## Future Improvements

* Cross-encoder reranking
* Query/result caching
* Incremental index updates
* Dockerized deployment
* Kubernetes/ECR deployment
* Managed vector database
* More diverse production-style evaluation queries

