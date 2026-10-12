"""Load service manuals, split them into chunks and build the FAISS index."""
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from . import config
from .vectorstore import Chunk, VectorStore

HEADING = re.compile(r"^\s*(#{1,6}\s+.+|\d+(\.\d+)*\s+[A-Z].{2,80}|[A-Z][A-Z0-9 /&-]{4,80})\s*$")


def load_pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(None, path.read_text(encoding="utf-8", errors="ignore"))]


def _split_long(paragraph: str, size: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts, current = [], ""
    for s in sentences:
        while len(s) > size:  # pathological sentence with no punctuation
            parts.append(s[:size])
            s = s[size:]
        if current and len(current) + len(s) + 1 > size:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    """Yield (section, chunk_text). Packs paragraphs up to `size`, carrying `overlap` chars forward."""
    section: str | None = None
    current, current_section, has_body = "", None, False
    for raw in re.split(r"\n\s*\n", text):
        paragraph = raw.strip()
        if not paragraph:
            continue
        first_line = paragraph.splitlines()[0]
        is_heading_only = bool(HEADING.match(first_line)) and "\n" not in paragraph
        if HEADING.match(first_line):
            if current and has_body:
                yield current_section, current
            current, has_body = "", False
            section = first_line.lstrip("#").strip()
        has_body = has_body or not is_heading_only
        for piece in _split_long(paragraph, size):
            if current and len(current) + len(piece) + 2 > size:
                yield current_section, current
                tail = current[-overlap:] if overlap else ""
                current = f"{tail}\n\n{piece}" if tail else piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
            current_section = section
    if current and has_body:
        yield current_section, current


def collect_chunks(manuals_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(manuals_dir.rglob("*")):
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            continue
        for page, text in load_pages(path):
            for section, body in chunk_text(text):
                chunks.append(Chunk(f"c{len(chunks)}", path.name, page, section, body))
    return chunks


def build_index(
    manuals_dir: Path = config.MANUALS_DIR,
    index_dir: Path = config.INDEX_DIR,
    embed: Callable[[list[str]], np.ndarray] | None = None,
) -> VectorStore:
    if embed is None:
        from .llm import embed
    chunks = collect_chunks(manuals_dir)
    if not chunks:
        raise ValueError(f"No {sorted(config.SUPPORTED_EXTENSIONS)} manuals found in {manuals_dir}")
    # Prefix the section heading so short chunks still carry their context.
    vectors = embed([f"{c.section}\n{c.text}" if c.section else c.text for c in chunks])
    store = VectorStore.build(chunks, vectors)
    store.save(index_dir)
    return store
