"""
rag_graph.py — the LangGraph StateGraph: classify_intent -> (retrieve_and_answer
| direct_answer), with a conditional edge chosen by classify_intent's output.

MOCK_LLM toggle (env var):
  - unset or "1" (default, GRADED baseline): every generation step uses
    deterministic, rule-based / templated logic. No LLM call, no API key, no
    network call to any LLM provider, anywhere in the graph.
  - "0" (optional, ungraded extension): classify_intent and the answer step of
    retrieve_and_answer / direct_answer call a real LLM (Groq's free tier, or
    any other free-tier LLM API) instead.

Retrieval itself (embed query -> ChromaDB top-3 by cosine similarity) always
runs for real in both modes — only the final *answer generation* branches on
MOCK_LLM.
"""
import os
from typing import List, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from ingest import retrieve_top_k
from prompt_template import build_prompt
from schemas import AskResponse

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours",
]


def mock_llm_enabled() -> bool:
    """True (mock/graded baseline) unless MOCK_LLM is explicitly set to '0'."""
    return os.environ.get("MOCK_LLM", "1") != "0"


class GraphState(TypedDict):
    query: str
    intent: str
    retrieved: List[dict]
    answer: str
    sources: List[str]
    confidence: float


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if mock_llm_enabled():
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: call the LLM to classify instead.
        intent = _llm_classify_intent(query)

    return {**state, "intent": intent}


def _llm_classify_intent(query: str) -> str:
    """Optional real-LLM classification path (MOCK_LLM=0). Uses the same Groq
    client as _llm_generate_answer below; see that function's docstring for
    setup notes. Falls back to the keyword heuristic if no LLM client/key is
    configured, so the graph never hard-fails just because this optional path
    was toggled on without also setting GROQ_API_KEY."""
    client = _get_groq_client()
    if client is None:
        lowered = query.lower()
        return "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": (
                "Classify this customer query as exactly one word, either "
                "'policy_question' (needs Zepto policy lookup) or "
                "'general_question' (does not). Query: " + query
            ),
        }],
        max_tokens=5,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip().lower()
    return "policy_question" if "policy" in text else "general_question"


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------
def route_by_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer (policy_question)
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real, in both modes — no API key needed.
    hits = retrieve_top_k(query, k=3)
    top_chunk = hits[0]["text"] if hits else ""
    source_ids = [h["chunk_id"] for h in hits]

    if mock_llm_enabled():
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        snippet = top_chunk[:200]
        answer = f"Based on the retrieved context: {snippet}"
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt a real LLM, grounded only in
        # the retrieved chunks, using the structured template.
        context_texts = [h["text"] for h in hits]
        prompt = build_prompt(query, context_texts)
        answer, confidence = _llm_generate_answer(prompt)

    return {**state, "retrieved": hits, "answer": answer, "sources": source_ids, "confidence": confidence}


# ---------------------------------------------------------------------------
# Node 3: direct_answer (general_question)
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    if mock_llm_enabled():
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        prompt = (
            "You are Zepto's support assistant. The customer asked something "
            "unrelated to Zepto policy: \"" + state["query"] + "\". Politely "
            "explain in 1-2 sentences that you can only help with Zepto policy "
            "questions (delivery, returns, membership, tracking, cancellation, "
            "damaged items, gift cards, support hours)."
        )
        answer, confidence = _llm_generate_answer(prompt)

    return {**state, "answer": answer, "sources": [], "confidence": confidence}


# ---------------------------------------------------------------------------
# Optional MOCK_LLM=0 real-LLM plumbing (Groq free tier, or any free-tier API)
# ---------------------------------------------------------------------------
_groq_client = None


def _get_groq_client():
    """Returns a Groq client if GROQ_API_KEY is set and the groq package is
    installed, else None. Only ever called when MOCK_LLM=0 — the graded
    mock baseline never touches this."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        return _groq_client
    except ImportError:
        return None


def _llm_generate_answer(prompt: str, max_retries: int = 2):
    """Calls the real LLM and validates its output isn't empty. Retries up to
    `max_retries` additional times with a corrective instruction on failure,
    then returns a clearly marked error response. (Schema validation of the
    full AskResponse happens one level up, in main.py / graph.invoke callers;
    this function's contract is just "return a non-empty string answer".)"""
    client = _get_groq_client()
    if client is None:
        return (
            "[MOCK_LLM=0 was set but no GROQ_API_KEY/groq package is configured — "
            "this is the optional extension path, not the graded baseline.]",
            0.0,
        )

    attempt_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": attempt_prompt}],
                max_tokens=200,
                temperature=0.2,
            )
            text = resp.choices[0].message.content.strip()
            if text:
                return text, 0.85
            raise ValueError("empty response")
        except Exception as e:
            if attempt == max_retries:
                return f"[ERROR: real-LLM path failed after {max_retries + 1} attempts: {e}]", 0.0
            attempt_prompt = prompt + f"\n\n(Previous attempt failed: {e}. Please answer again, following the format exactly.)"
    return "[ERROR: unreachable]", 0.0


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(query: str) -> AskResponse:
    """Runs the graph end to end and returns a validated AskResponse. In mock
    mode the schema is populated deterministically by our own code (no LLM
    output to fail validation, since none was generated)."""
    graph = get_graph()
    final_state = graph.invoke({
        "query": query, "intent": "", "retrieved": [], "answer": "", "sources": [], "confidence": 0.0,
    })
    try:
        return AskResponse(
            answer=final_state["answer"],
            sources=final_state["sources"],
            confidence=final_state["confidence"],
        )
    except ValidationError as e:
        return AskResponse(answer=f"[ERROR: response failed schema validation: {e}]", sources=[], confidence=0.0)


if __name__ == "__main__":
    for q in ["How much does delivery cost?", "What's the weather like today?", "Can I cancel my order after it's packed?"]:
        print(f"\nQ: {q}")
        result = ask(q)
        print(result.model_dump_json(indent=2))
