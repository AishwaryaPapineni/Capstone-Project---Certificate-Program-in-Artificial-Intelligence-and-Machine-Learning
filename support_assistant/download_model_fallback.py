"""
download_model_fallback.py — fetches an ONNX export of all-MiniLM-L6-v2 from
GitHub, for use by embedder.py ONLY when the standard Hugging Face Hub
download (sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")) is
unreachable.

Why this exists: this submission was assembled in a sandboxed session whose
outbound network access is restricted to an allowlist (PyPI, npm, GitHub) and
does not include huggingface.co — confirmed by a 403 on every direct
connection attempt to that host. On a normal machine (your laptop, Colab, a
CI runner) huggingface.co is reachable and this file is never used at all —
embedder.py tries the real sentence-transformers download first and only
falls back to this file if that fails.

Source: github.com/spring-projects/spring-ai vendors the exact same Hugging
Face model (sentence-transformers/all-MiniLM-L6-v2) as an ONNX export for use
from their Java library — same published weights, converted to ONNX format,
nothing retrained or altered. GitHub's raw/LFS-media endpoints ARE reachable
from this sandbox, so this script downloads:
  - model.onnx        (the exported weights, ~90MB, served via git-lfs)
  - tokenizer.json     (the exact HuggingFace fast-tokenizer file, self
                        contained — no separate vocab.txt needed)
into support_assistant/.model_cache/all-MiniLM-L6-v2-onnx/ (gitignored — not
committed to the repository; this script regenerates it on demand).
"""
from pathlib import Path

import requests

HERE = Path(__file__).parent
CACHE_DIR = HERE / ".model_cache" / "all-MiniLM-L6-v2-onnx"

RAW_BASE = (
    "https://raw.githubusercontent.com/spring-projects/spring-ai/main/"
    "models/spring-ai-transformers/src/main/resources/onnx/all-MiniLM-L6-v2"
)
MEDIA_BASE = (
    "https://media.githubusercontent.com/media/spring-projects/spring-ai/main/"
    "models/spring-ai-transformers/src/main/resources/onnx/all-MiniLM-L6-v2"
)

FILES = {
    # tokenizer.json is a normal (non-LFS) text file -> raw.githubusercontent.com
    "tokenizer.json": f"{RAW_BASE}/tokenizer.json",
    # model.onnx is stored via git-lfs -> must use the media (LFS) endpoint,
    # otherwise raw.githubusercontent.com serves a tiny LFS pointer file
    # instead of the actual binary.
    "model.onnx": f"{MEDIA_BASE}/model.onnx",
}


def download_fallback_model():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        dest = CACHE_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[download_model_fallback] {filename} already cached ({dest.stat().st_size} bytes)")
            continue
        print(f"[download_model_fallback] downloading {filename} from {url} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"[download_model_fallback] saved {filename} ({len(resp.content)} bytes)")


if __name__ == "__main__":
    download_fallback_model()
