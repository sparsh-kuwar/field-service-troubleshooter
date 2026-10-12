import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: int | None
    section: str | None
    text: str


class VectorStore:
    """FAISS inner-product index over L2-normalised vectors (i.e. cosine similarity)."""

    def __init__(self, index: faiss.Index, chunks: list[Chunk]):
        self.index = index
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: np.ndarray) -> "VectorStore":
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index, chunks)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "manuals.faiss"))
        (directory / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks], indent=1))

    @classmethod
    def load(cls, directory: Path) -> "VectorStore | None":
        index_path, chunks_path = directory / "manuals.faiss", directory / "chunks.json"
        if not (index_path.exists() and chunks_path.exists()):
            return None
        chunks = [Chunk(**c) for c in json.loads(chunks_path.read_text())]
        return cls(faiss.read_index(str(index_path)), chunks)

    def search(self, query_vectors: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Search with several query vectors; keep each chunk's best score."""
        scores, ids = self.index.search(query_vectors, min(k, len(self.chunks)))
        best: dict[int, float] = {}
        for row_scores, row_ids in zip(scores, ids):
            for score, idx in zip(row_scores, row_ids):
                if idx >= 0 and score > best.get(idx, -1.0):
                    best[idx] = float(score)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(self.chunks[i], s) for i, s in ranked]
