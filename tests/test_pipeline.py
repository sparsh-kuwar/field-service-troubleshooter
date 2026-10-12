"""Offline tests: a hashed bag-of-words embedder and scripted chat replace OpenAI."""
import re
import zlib
from pathlib import Path

import faiss
import numpy as np
import pytest

from app.agents import Orchestrator
from app.ingest import build_index, chunk_text

MANUALS = Path(__file__).resolve().parent.parent / "manuals"


def fake_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), 512), dtype="float32")
    for i, t in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, zlib.crc32(word.encode()) % 512] += 1
    faiss.normalize_L2(out)
    return out


def make_chat(diagnosis: dict, queries=("ignition failure igniter gas pressure",)):
    def chat(system: str, user: str) -> dict:
        if "triage agent" in system:
            return {"equipment": "TF-200", "symptoms": ["no flame"], "error_codes": [],
                    "safety_concerns": [], "search_queries": list(queries)}
        if "evidence grader" in system:
            ids = re.findall(r"^\[(c\d+)\]", user, re.M)
            return {"relevant_ids": ids[:2], "rationale": "test"}
        return diagnosis
    return chat


@pytest.fixture
def store(tmp_path):
    return build_index(MANUALS, tmp_path, embed=fake_embed)


def test_chunker_respects_size_and_tracks_sections():
    text = "# Intro\n\n" + "word " * 400 + "\n\n# Next\n\nshort paragraph"
    chunks = list(chunk_text(text, size=300, overlap=50))
    assert all(len(body) <= 300 + 50 + 2 for _, body in chunks)
    assert chunks[-1][0] == "Next"


def test_resolved_answer_keeps_only_cited_items(store):
    diagnosis = {
        "summary": "Ignition failure.",
        "likely_causes": [{"cause": "Low gas pressure", "citations": ["c3"]},
                          {"cause": "Invented cause", "citations": ["c999"]}],
        "steps": [{"instruction": "Check gas valve", "citations": ["c3"]},
                  {"instruction": "Uncited step", "citations": []}],
        "safety_warnings": [], "confidence": "high", "insufficient_evidence": False,
    }
    # Make the diagnosis cite whatever the grader actually kept.
    def chat(system, user):
        if "diagnosis agent" in system:
            first = re.findall(r"^\[(c\d+)\]", user, re.M)[0]
            for item in diagnosis["likely_causes"][:1] + diagnosis["steps"][:1]:
                item["citations"] = [first]
            return diagnosis
        return make_chat(diagnosis)(system, user)

    result = Orchestrator(store, fake_embed, chat, min_score=0.05, escalation_score=0.1).troubleshoot(
        "E1 ignition failure, igniter clicks but no flame, check gas pressure")
    assert result.status == "resolved"
    assert [c["cause"] for c in result.diagnosis["likely_causes"]] == ["Low gas pressure"]
    assert [s["instruction"] for s in result.diagnosis["steps"]] == ["Check gas valve"]
    assert result.trace[-1].detail["dropped_uncited_items"] == 2


def test_low_similarity_escalates_without_diagnosis(store):
    def chat(system, user):
        assert "diagnosis agent" not in system and "evidence grader" not in system
        return make_chat({}, queries=("conveyor belt hydraulic pump",))(system, user)

    result = Orchestrator(store, fake_embed, chat, min_score=0.2, escalation_score=0.6).troubleshoot(
        "forklift hydraulic pump whining")
    assert result.status == "escalated"
    assert result.diagnosis is None
    assert "threshold" in result.escalation["reason"]


def test_insufficient_evidence_escalates(store):
    chat = make_chat({"summary": "", "steps": [], "insufficient_evidence": True})
    result = Orchestrator(store, fake_embed, chat, min_score=0.05, escalation_score=0.1).troubleshoot(
        "E1 ignition failure igniter gas pressure")
    assert result.status == "escalated"
    assert result.escalation["closest_manual_sections"]
