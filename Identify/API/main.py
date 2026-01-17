from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import ResolveSourcesRequest, ResolveSourcesResponse
from .geminiSourceResolver import GeminiSourceResolver

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_resolver: GeminiSourceResolver | None = None

def get_resolver() -> GeminiSourceResolver:
    global _resolver
    if _resolver is None:
        _resolver = GeminiSourceResolver()
    return _resolver


@app.post("/api/resolve-sources", response_model=ResolveSourcesResponse)
def resolve_sources_endpoint(req: ResolveSourcesRequest):
    try:
        resolver = get_resolver()
        sources = resolver.resolve_sources(req.query)
        return {"query": req.query, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
