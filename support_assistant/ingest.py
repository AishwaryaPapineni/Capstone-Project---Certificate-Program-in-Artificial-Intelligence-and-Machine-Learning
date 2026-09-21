"""
ingest.py — loads the 8 policy documents, chunks them, embeds each chunk with
all-MiniLM-L6-v2 (sentence-transformers, local/free/no API key), and stores
the embeddings in a persistent ChromaDB collection on disk (./chroma_db).

Run directly to (re)build the index:
    python ingest.py

Imported by rag_graph.py, which reuses get_collection() for retrieval.
"""
from pathlib import Path

import chromadb

from embedder import embed_texts

HERE = Path(__file__).parent
DOCS_DIR = HERE / "docs"
CHROMA_PATH = HERE / "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 220  # characters; each policy doc is short, so one paragraph is
CHUNK_OVERLAP = 40  # already close to one "chunk" — this fixed-size scheme
# additionally splits any doc that runs long, with a bit of overlap so a
# sentence isn't awkwardly cut in half between two chunks.


def chunk_text(text: str, doc_id: str):
    """Simple fixed-size character chunking with overlap. Short docs (which is
    all 8 of ours) naturally come back as a single chunk."""
    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [{"chunk_id": f"{doc_id}_chunk0", "text": text}]
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append({"chunk_id": f"{doc_id}_chunk{idx}", "text": text[start:end]})
        idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_documents():
    docs = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        doc_id = path.stem  # e.g. "doc_01"
        text = path.read_text(encoding="utf-8").strip()
        docs.append({"doc_id": doc_id, "text": text})
    return docs


def build_index(reset: bool = True):
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    # hnsw:space="cosine" makes ChromaDB rank results by cosine similarity
    # (the assignment's required retrieval metric), rather than its default L2.
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    docs = load_documents()

    all_ids, all_texts, all_metas = [], [], []
    for doc in docs:
        for chunk in chunk_text(doc["text"], doc["doc_id"]):
            all_ids.append(chunk["chunk_id"])
            all_texts.append(chunk["text"])
            all_metas.append({"doc_id": doc["doc_id"]})

    embeddings = embed_texts(all_texts).tolist()
    collection.add(ids=all_ids, documents=all_texts, embeddings=embeddings, metadatas=all_metas)

    print(f"Indexed {len(all_ids)} chunks from {len(docs)} documents into '{COLLECTION_NAME}' at {CHROMA_PATH}")
    return collection


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(COLLECTION_NAME)


def retrieve_top_k(query: str, k: int = 3):
    collection = get_collection()
    query_embedding = embed_texts([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


if __name__ == "__main__":
    build_index(reset=True)
    print("\nSanity check — retrieving top-3 for 'how long do I have to return a damaged item?':")
    for hit in retrieve_top_k("how long do I have to return a damaged item?", k=3):
        print(f"  [{hit['chunk_id']}] dist={hit['distance']:.4f}  {hit['text'][:90]}...")
