from typing import List
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Answer to the user query")
    sources: List[str] = Field(default_factory=list, description="List of document chunk IDs used as context")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")