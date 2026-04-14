"""
FastAPI backend for the market research assistant.

Endpoints:
    GET  /           — serves the web UI
    POST /query      — runs the Scout → Mapper → Critic pipeline

Usage:
    uvicorn app.main:app --reload
"""

import pathlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.orchestrator import run_pipeline

app = FastAPI(title="Market Research Analyst")

STATIC_DIR = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    query: str


class ThemeResult(BaseModel):
    theme_name: str
    companies: list[str]
    rationale: str
    citations: list[str]


class QueryResponse(BaseModel):
    query: str
    themes: list[ThemeResult]


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        reviewed_map = run_pipeline(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        query=request.query,
        themes=[ThemeResult(**theme) for theme in reviewed_map],
    )
