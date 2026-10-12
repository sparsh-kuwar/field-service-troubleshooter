import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config, llm
from app.agents import Orchestrator
from app.ingest import build_index
from app.vectorstore import VectorStore

log = logging.getLogger("uvicorn.error")
state: dict = {"store": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["store"] = VectorStore.load(config.INDEX_DIR)
    # Ephemeral disks (e.g. Render free tier) lose the index on every restart:
    # rebuild it from the manuals committed in the repo before serving requests.
    _ensure_index()
    yield


def _ensure_index():
    """Load or build the index if it is missing. Returns the store, or None if it can't be built."""
    if state["store"] is None and os.getenv("OPENAI_API_KEY"):
        try:
            state["store"] = build_index(config.MANUALS_DIR, config.INDEX_DIR)
            log.info("Built index with %d chunks from %s", len(state["store"].chunks), config.MANUALS_DIR)
        except Exception as e:  # keep serving; retried on the next request
            log.warning("Could not build index: %s", e)
    return state["store"]


app = FastAPI(title="Field-Service Troubleshooting Assistant", lifespan=lifespan)


class TroubleshootRequest(BaseModel):
    question: str = Field(min_length=3, examples=["Unit shows E1 and won't light, clicking but no flame"])
    equipment: str | None = Field(default=None, examples=["ThermaFlow TF-200"])
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    if config.ADMIN_TOKEN and not secrets.compare_digest(x_admin_token or "", config.ADMIN_TOKEN):
        raise HTTPException(403, "Re-indexing and uploads are disabled on this deployment.")


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
        "admin_locked": config.ADMIN_TOKEN is not None,
    }


@app.post("/ingest", dependencies=[Depends(require_admin)])
def ingest():
    """Re-embed every manual in the manuals directory and rebuild the FAISS index."""
    return _rebuild()


@app.post("/manuals", dependencies=[Depends(require_admin)])
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
    store = _ensure_index()
    if store is None:
        raise HTTPException(
            503,
            "The service manuals are still being prepared, or the server isn't configured yet. "
            "Please try again in a minute; if this keeps happening, the administrator needs to set OPENAI_API_KEY.",
        )
    try:
        result = Orchestrator(store, llm.embed, llm.chat_json).troubleshoot(req.question, req.equipment, req.top_k)
    except llm.LLMUnavailable as e:
        raise HTTPException(503, str(e))
    return result.to_dict()
