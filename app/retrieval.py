import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

_index = None
_meta = None
_model = None

def load():
    global _index, _meta, _model
    if _index is None:
        print("Loading model and index...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _index = faiss.read_index("artifacts/faiss.index")
        with open("artifacts/metadata.json") as f:
            _meta = json.load(f)
        print("Ready.")
    return _index, _meta, _model

def search(text: str, k=5):
    index, meta, model = load()
    vec = model.encode([text], convert_to_numpy=True).astype("float32")
    D, I = index.search(vec, k)
    return [meta[idx] for idx in I[0]]
