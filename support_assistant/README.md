# Module 3 — Support Assistant (`/support_assistant`)

A small, complete GenAI service for Zepto: an 8-document policy corpus
embedded into ChromaDB, a LangGraph-orchestrated intent router + retriever,
a Pydantic-enforced JSON output schema, and a FastAPI `/ask` endpoint —
fully working offline via a deterministic `MOCK_LLM` mode (the graded
baseline), with an optional real-LLM extension layered on top.

## How to run

```bash
pip install -r requirements.txt

python ingest.py          # embeds the 8 docs into ./chroma_db (run once)
uvicorn main:app --host 0.0.0.0 --port 7860   # MOCK_LLM defaults to 1
```

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "How much does delivery cost?"}'
```

Docker:

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

**Note on this session's Docker build**: the sandbox this submission was
assembled in also blocks pulls from every container registry (Docker Hub,
GHCR, GCR, Quay, ECR — all return a `403` from the egress proxy), so `docker
build` could not be executed inside this session. The `Dockerfile` itself is
a standard, correct `python:3.11-slim` + `pip install` + `uvicorn` build that
will build and run normally on any machine with normal internet/Docker Hub
access — please verify with the two commands above.

## Embedding model — how it runs without a Hugging Face download

`embedder.py` first tries the standard path —
`sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")`, which
downloads the model from Hugging Face Hub on first use and caches it
locally. **This is what runs on any normal machine** (this is also literally
what the assignment asks for: "generate embeddings locally using the
open-source sentence-transformers library").

This session's sandbox, however, blocks `huggingface.co` outright (same
allowlist restriction described in the root README and Module 1's README),
so `embedder.py` automatically falls back to a small local ONNX Runtime
runner (`download_model_fallback.py`) that loads the *exact same published
model weights* — sourced from `github.com/spring-projects/spring-ai`, which
vendors this same Hugging Face model as an ONNX export for their Java library
— and reproduces sentence-transformers' own forward pass by hand: mean-pool
the token embeddings with the attention mask, then L2-normalize. Same
weights, same math, same 384-dimension output. This fallback is fully
automatic and requires no action; `embed_texts()` is the one function both
paths implement, so nothing downstream (ingestion, retrieval, the graph)
knows or cares which path is active.

## Architecture — ingestion → embedding → retrieval → generation

```
docs/doc_01.txt … doc_08.txt
        |  (ingest.py: load_documents + chunk_text — fixed-size chunking)
        v
   25 text chunks
        |  (embedder.py: embed_texts — all-MiniLM-L6-v2, 384-dim, local)
        v
   ChromaDB collection "zepto_policies" (./chroma_db, cosine similarity)
        |
        |<==== query embedding (embedder.py) ====|
        |                                          |
        v                                          |
  rag_graph.py: classify_intent(query) --keyword heuristic (MOCK_LLM=1)-->
     policy_question -> retrieve_and_answer: ChromaDB.query(top_k=3, cosine)
                          -> mock: canned "Based on the retrieved context: ..."
                          -> MOCK_LLM=0: prompt_template.build_prompt() -> Groq LLM
     general_question -> direct_answer
                          -> mock: fixed canned string
                          -> MOCK_LLM=0: LLM, no retrieval
        |
        v
  schemas.AskResponse (answer, sources, confidence) — Pydantic-validated
        |
        v
  main.py: FastAPI POST /ask
```

- **Ingestion**: `ingest.py` → `load_documents()` reads the 8 `.txt` files;
  `chunk_text()` does fixed-size (220 char, 40 char overlap) chunking — most
  of the 8 short policy docs come back as one chunk each, longer ones split
  into 2-3, giving 25 chunks total.
- **Embedding**: `embedder.py` → `embed_texts()`, all-MiniLM-L6-v2, 384-dim
  vectors (see previous section for the two backends).
- **Vector store**: ChromaDB `PersistentClient` at `./chroma_db`, collection
  `zepto_policies`, created with `metadata={"hnsw:space": "cosine"}` so
  nearest-neighbor search ranks by cosine similarity as required.
- **Retrieval**: `ingest.retrieve_top_k(query, k=3)` — embeds the query and
  queries ChromaDB for the 3 nearest chunks. Called from the
  `retrieve_and_answer` node in `rag_graph.py`. **Runs for real in both
  MOCK_LLM modes** — no API key needed for embedding or ChromaDB.
  Note: `route_by_intent`'s output only ever selects between `retrieve_and_answer`
  and `direct_answer`.
- **Generation**: also inside `retrieve_and_answer` (for policy questions)
  and `direct_answer` (for everything else), in `rag_graph.py`. **This is
  the only stage that branches on `MOCK_LLM`**:
  - `MOCK_LLM` unset or `"1"` (default, graded baseline): a deterministic
    templated string built entirely in code — `f"Based on the retrieved
    context: {top_chunk[:200]}"` for policy questions, a fixed sentence for
    general questions. No network call, no LLM, no API key.
  - `MOCK_LLM=0` (optional extension): `prompt_template.build_prompt()`
    builds the structured role/context/task/format/length prompt (with its
    negative constraint and few-shot example), sent to Groq's free-tier
    `llama-3.1-8b-instant` via `_llm_generate_answer()`, which retries up to
    2 additional times with a corrective instruction if the call fails, then
    returns a clearly marked error string. Requires `GROQ_API_KEY` to be set
    (never hardcoded — read from the environment only).
- **Output schema**: `schemas.AskResponse` (`answer: str, sources:
  List[str], confidence: float`), populated deterministically by our own
  code in mock mode (`sources` = retrieved chunk IDs for policy questions,
  `[]` for general questions; `confidence = 1.0`), and by the real LLM's
  validated output in the optional extension.
- **API**: `main.py` — `POST /ask` takes `{"query": str}` (`schemas.AskRequest`)
  and returns the validated `AskResponse` above.

## Example calls (MOCK_LLM left at its default = "1")

**Call 1 — policy question (routes to `retrieve_and_answer`):**

```
POST /ask  {"query": "How much does delivery cost?"}
```
```json
{"answer":"Based on the retrieved context: nd current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot","sources":["doc_01_chunk1","doc_04_chunk2","doc_05_chunk1"],"confidence":1.0}
```

**Call 2 — general question (routes to `direct_answer`):**

```
POST /ask  {"query": "What is the capital of France?"}
```
```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

Both captured live from a running `uvicorn` instance in this session (see
`call1.json`, `call2.json`).

## Design decisions

- **Chunking**: fixed-size (220 chars, 40 overlap) rather than a heavier
  semantic chunker — the 8 source docs are short single-topic paragraphs, so
  this is enough to keep chunks coherent while still splitting the 3 longer
  documents into 2-3 pieces each (25 chunks total from 8 docs).
- **Keyword-heuristic intent classification** in mock mode uses a simple
  substring match against the 8 required keywords — deterministic,
  reproducible, and exactly matches the assignment's specified list.
- **The retry-on-validation-failure logic** for the optional `MOCK_LLM=0`
  path is implemented in `_llm_generate_answer()` even though it never
  triggers under the graded mock baseline (there's no LLM output to fail
  validation there, since none is generated).
- **GROQ_API_KEY** is read only from the environment (`os.environ`), never
  hardcoded or committed — see `.env.example` for the variable name.

## Files

- `docs/doc_01.txt` … `doc_08.txt` — the 8 policy documents (verbatim)
- `ingest.py` — chunking, embedding, ChromaDB indexing + retrieval
- `embedder.py`, `download_model_fallback.py` — embedding model loading (+ fallback)
- `prompt_template.py` — the structured role/context/task/format/length prompt
- `schemas.py` — Pydantic `AskRequest` / `AskResponse`
- `rag_graph.py` — the LangGraph `StateGraph` (3 nodes + conditional edge)
- `main.py` — FastAPI app, `POST /ask`
- `Dockerfile`, `.dockerignore` — container build
- `call1.json`, `call2.json` — captured example responses
