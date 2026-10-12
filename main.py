from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config, llm
from app.agents import Orchestrator
from app.ingest import build_index
from app.vectorstore import VectorStore

state: dict = {"store": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["store"] = VectorStore.load(config.INDEX_DIR)
    yield


app = FastAPI(title="Field-Service Troubleshooting Assistant", lifespan=lifespan)


class TroubleshootRequest(BaseModel):
    question: str = Field(min_length=3, examples=["Unit shows E1 and won't light, clicking but no flame"])
    equipment: str | None = Field(default=None, examples=["ThermaFlow TF-200"])
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


def _rebuild() -> dict:
    try:
        store = build_index(config.MANUALS_DIR, config.INDEX_DIR)
    except llm.LLMUnavailable as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    state["store"] = store
    return {"chunks": len(store.chunks), "sources": sorted({c.source for c in store.chunks})}


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    store = state["store"]
    return {
        "index_loaded": store is not None,
        "chunks": len(store.chunks) if store else 0,
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


@app.post("/ingest")
def ingest():
    """Re-embed every manual in the manuals directory and rebuild the FAISS index."""
    return _rebuild()


@app.post("/manuals")
def upload_manual(file: UploadFile):
    """Upload a PDF/TXT/MD manual, then rebuild the index."""
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Supported types: {sorted(config.SUPPORTED_EXTENSIONS)}")
    config.MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    (config.MANUALS_DIR / name).write_bytes(file.file.read())
    return {"saved": name, **_rebuild()}


@app.post("/troubleshoot")
def troubleshoot(req: TroubleshootRequest):
    store = state["store"]
    if store is None:
        raise HTTPException(409, "No index yet. POST /ingest first.")
    try:
        result = Orchestrator(store, llm.embed, llm.chat_json).troubleshoot(req.question, req.equipment, req.top_k)
    except llm.LLMUnavailable as e:
        raise HTTPException(503, str(e))
    return result.to_dict()
