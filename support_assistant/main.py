"""
main.py — FastAPI wrapper around the LangGraph RAG pipeline.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 7860

Then:
    curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \\
         -d '{"query": "How much does delivery cost?"}'
"""
from fastapi import FastAPI

from rag_graph import ask
from schemas import AskRequest, AskResponse

app = FastAPI(title="Zepto Support Assistant", version="1.0")


@app.get("/")
def root():
    return {"status": "ok", "service": "zepto-support-assistant"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    return ask(request.query)
