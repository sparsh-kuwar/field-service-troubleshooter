import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MANUALS_DIR = Path(os.getenv("MANUALS_DIR", BASE_DIR / "manuals"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "index"))

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "6"))

# Cosine similarity thresholds (embeddings are L2-normalised, FAISS inner product).
# Chunks below MIN_CHUNK_SCORE are discarded; if the best chunk is below
# ESCALATION_SCORE the problem is escalated without asking the LLM to answer.
MIN_CHUNK_SCORE = float(os.getenv("MIN_CHUNK_SCORE", "0.25"))
ESCALATION_SCORE = float(os.getenv("ESCALATION_SCORE", "0.35"))

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
