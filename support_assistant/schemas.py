"""schemas.py — Pydantic models enforcing the API's request/response shape."""
from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list, description="chunk/document IDs used")
    confidence: float = Field(ge=0.0, le=1.0)
