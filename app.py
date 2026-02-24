"""RVR Answer Engine - FastAPI Application"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn

from rvr_engine import RVREngine
from document_store import DocumentStore

app = FastAPI(
    title="RVR Answer Engine",
    description="Retrieve-Verify-Retrieve for comprehensive question answering",
    version="1.0.0"
)

# Initialize document store and RVR engine
doc_store = DocumentStore()
doc_store.load_sample_documents()
rvr_engine = RVREngine(doc_store)


class SearchRequest(BaseModel):
    query: str = Field(..., description="Research question or query")
    max_rounds: int = Field(3, ge=1, le=5, description="Maximum retrieval rounds")
    top_k: int = Field(5, ge=1, le=20, description="Documents to retrieve per round")


class SearchResponse(BaseModel):
    query: str
    rounds: List[Dict[str, Any]]
    total_verified: int
    coverage_metrics: Dict[str, float]


@app.get("/")
async def root():
    return {
        "message": "RVR Answer Engine API",
        "paper": "https://arxiv.org/abs/2602.18425v1",
        "endpoints": {
            "/search": "POST - Perform RVR search",
            "/documents": "GET - List available documents",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "documents_loaded": len(doc_store.documents)}


@app.get("/documents")
async def list_documents():
    """Return summary of available documents"""
    return {
        "total_documents": len(doc_store.documents),
        "sample": doc_store.documents[:3]
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform RVR search with multi-round retrieval and verification"""
    try:
        result = rvr_engine.search(
            query=request.query,
            max_rounds=request.max_rounds,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting RVR Answer Engine...")
    print("API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
