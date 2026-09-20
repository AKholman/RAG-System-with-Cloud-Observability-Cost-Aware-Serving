import json
import faiss
from sentence_transformers import SentenceTransformer

_index = None
_meta = None
_model = None


def load():
    global _index, _meta, _model

    if _index is None:
        print("Loading embedding model and FAISS index...")

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _index = faiss.read_index("artifacts/faiss.index")

        with open("artifacts/metadata.json") as f:
            _meta = json.load(f)

        print(f"Loaded {_index.ntotal} vectors.")
        print("RAG retrieval ready.")

    return _index, _meta, _model


def search(text: str, k: int = 5):
    index, meta, model = load()

    vector = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    _, indices = index.search(vector, k)

    results = []

    for idx in indices[0]:
        if idx < 0:
            continue
        results.append(meta[idx])

    return results
