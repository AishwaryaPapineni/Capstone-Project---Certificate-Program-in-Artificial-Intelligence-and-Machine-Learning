"""
embedder.py — loads the all-MiniLM-L6-v2 sentence-embedding model and exposes
one function, embed_texts(texts) -> np.ndarray, used by ingest.py and
rag_graph.py for both document-chunk embedding and query embedding.

Primary path: sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2").
This is the standard, documented way to run this model, and is what runs on
any machine with normal internet access — the first call downloads the model
from Hugging Face Hub and caches it locally; every later run reuses the cache
offline, exactly like sns.load_dataset() in Module 2.

Fallback path (used automatically only if that download fails, e.g. in this
submission's network-restricted sandbox): a small local ONNX Runtime runner
that loads the *exact same model weights* (converted to ONNX) and reproduces
sentence-transformers' own forward pass by hand — mean-pool the token
embeddings using the attention mask, then L2-normalize — which is exactly
what SentenceTransformer("all-MiniLM-L6-v2").encode(...) does internally.
Same weights, same math, same 384-dim output; only the loading mechanism
differs. See download_model_fallback.py for exactly where those weights come
from and why (short version: a Spring AI project on GitHub vendors the same
Hugging Face model as an ONNX export for JVM use, and GitHub — unlike Hugging
Face — is reachable from this sandbox).
"""
from pathlib import Path

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
HERE = Path(__file__).parent
CACHE_DIR = HERE / ".model_cache" / "all-MiniLM-L6-v2-onnx"

_backend = None  # "sentence-transformers" | "onnx-fallback"
_st_model = None
_onnx_session = None
_tokenizer = None


def _try_load_sentence_transformers():
    global _st_model
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(MODEL_NAME)
        return True
    except Exception as e:
        print(f"[embedder] sentence-transformers direct download unavailable ({type(e).__name__}); "
              f"falling back to local ONNX runner. See download_model_fallback.py.")
        return False


def _ensure_fallback_files():
    from download_model_fallback import download_fallback_model
    if not (CACHE_DIR / "model.onnx").exists() or not (CACHE_DIR / "tokenizer.json").exists():
        download_fallback_model()


def _load_onnx_fallback():
    global _onnx_session, _tokenizer
    import onnxruntime as ort
    from tokenizers import Tokenizer

    _ensure_fallback_files()
    _tokenizer = Tokenizer.from_file(str(CACHE_DIR / "tokenizer.json"))
    _tokenizer.enable_padding()
    _tokenizer.enable_truncation(max_length=256)  # all-MiniLM-L6-v2's max_seq_length
    _onnx_session = ort.InferenceSession(str(CACHE_DIR / "model.onnx"))


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, a_min=1e-9, a_max=None)


def _embed_onnx_fallback(texts):
    encodings = _tokenizer.encode_batch(list(texts))
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    onnx_inputs = {inp.name: v for inp, v in zip(
        _onnx_session.get_inputs(),
        [input_ids, attention_mask, token_type_ids],
    )}
    outputs = _onnx_session.run(None, onnx_inputs)
    token_embeddings = outputs[0]  # (batch, seq_len, 384) — last_hidden_state
    pooled = _mean_pool(token_embeddings, attention_mask)
    return _l2_normalize(pooled)


def embed_texts(texts) -> np.ndarray:
    """Embed a list of strings, returns an (N, 384) float32 array."""
    global _backend
    if _backend is None:
        _backend = "sentence-transformers" if _try_load_sentence_transformers() else "onnx-fallback"
        if _backend == "onnx-fallback":
            _load_onnx_fallback()
        print(f"[embedder] using backend: {_backend}")

    if _backend == "sentence-transformers":
        return _st_model.encode(list(texts), convert_to_numpy=True)
    return _embed_onnx_fallback(texts)


if __name__ == "__main__":
    vecs = embed_texts(["a test sentence", "another one about Zepto delivery"])
    print("shape:", vecs.shape, "norm of row 0:", np.linalg.norm(vecs[0]))
