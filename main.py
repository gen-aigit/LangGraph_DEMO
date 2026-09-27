"""FastAPI entry point. Compiles and invokes the LangGraph workflow."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import openai  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.graph import build_graph  # noqa: E402


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    response: str
    intent_type: str | None = None
    domain: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The graph is compiled here, in the API layer, once at startup.
    app.state.graph = build_graph().compile()
    yield


app = FastAPI(title="Support Router", lifespan=lifespan)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    if not payload.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")

    try:
        print("Calling langraph")
        result = await request.app.state.graph.ainvoke({"query": payload.query})
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="LLM rate limit reached, try again shortly")
    except openai.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e.message}")
    except openai.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach the LLM service")

    return ChatResponse(
        query=payload.query,
        response=result["response"],
        intent_type=result.get("intent_type"),
        domain=result.get("domain"),
    )


def start() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)


if __name__ == "__main__":
    start()
