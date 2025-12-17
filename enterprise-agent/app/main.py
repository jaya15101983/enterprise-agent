"""
FastAPI application entry point.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os

from app.guardrails.pipeline import create_default_pipeline
from app.observability.tracing import setup_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_tracing()
    yield
    # Shutdown


app = FastAPI(
    title="Enterprise Agent",
    version="0.1.0",
    lifespan=lifespan
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str


# Initialize guardrails
guardrail_pipeline = create_default_pipeline()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat request through the agent pipeline."""
    
    # Run guardrails
    passed, results = await guardrail_pipeline.evaluate(
        request.message, 
        {"endpoint": "chat"}
    )
    
    if not passed:
        blocking = [r for r in results if r.action == "block"]
        if blocking:
            raise HTTPException(
                status_code=400, 
                detail=blocking[0].reason
            )
    
    # TODO: Integrate with agent graph
    # For now, return placeholder
    return ChatResponse(
        response="Agent response placeholder",
        thread_id=request.thread_id or "new-thread"
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check - verify dependencies."""
    # TODO: Check database, LLM connectivity
    return {"status": "ready"}
