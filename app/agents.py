"""Multi-agent troubleshooting workflow.

Triage -> Retrieval -> Evidence grading -> Diagnosis, with escalation whenever the
retrieved manual content does not clearly support an answer.
"""
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from . import config
from .vectorstore import VectorStore

EmbedFn = Callable[[list[str]], np.ndarray]
ChatFn = Callable[[str, str], dict]


@dataclass
class Evidence:
    chunk_id: str
    source: str
    page: int | None
    section: str | None
    text: str
    score: float


@dataclass
class TraceStep:
    agent: str
    detail: dict


@dataclass
class Result:
    status: str  # "resolved" | "escalated"
    question: str
    triage: dict
    diagnosis: dict | None = None
    escalation: dict | None = None
    evidence: list[Evidence] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _format_evidence(evidence: list[Evidence]) -> str:
    blocks = []
    for e in evidence:
        where = e.source + (f", p.{e.page}" if e.page else "") + (f", {e.section}" if e.section else "")
        blocks.append(f"[{e.chunk_id}] ({where})\n{e.text}")
    return "\n\n---\n\n".join(blocks)


class TriageAgent:
    SYSTEM = """You are the triage agent for a field-service troubleshooting assistant.
Extract structured facts from the technician's report. Do not diagnose.
Return JSON with keys:
  "equipment": string or null (make/model/type if mentioned),
  "symptoms": list of short strings,
  "error_codes": list of strings exactly as reported,
  "safety_concerns": list of hazards implied by the report (gas smell, burning, sparks, water near electrics...), empty if none,
  "search_queries": 1-3 short queries worded the way a service manual would describe this problem."""

    def __init__(self, chat: ChatFn):
        self.chat = chat

    def run(self, question: str, equipment: str | None) -> dict:
        user = f"Technician report: {question}" + (f"\nEquipment: {equipment}" if equipment else "")
        out = self.chat(self.SYSTEM, user)
        queries = [q for q in out.get("search_queries", []) if isinstance(q, str) and q.strip()][:3]
        return {
            "equipment": out.get("equipment") or equipment,
            "symptoms": list(out.get("symptoms", [])),
            "error_codes": list(out.get("error_codes", [])),
            "safety_concerns": list(out.get("safety_concerns", [])),
            "search_queries": queries,
        }


class RetrievalAgent:
    def __init__(self, store: VectorStore, embed: EmbedFn):
        self.store = store
        self.embed = embed

    def run(self, question: str, triage: dict, k: int) -> list[Evidence]:
        queries = [question, *triage["search_queries"]]
        if triage["error_codes"]:
            queries.append(" ".join(f"error code {c}" for c in triage["error_codes"]))
        hits = self.store.search(self.embed(queries), k)
        return [Evidence(c.chunk_id, c.source, c.page, c.section, c.text, round(s, 4)) for c, s in hits]


class EvidenceGrader:
    SYSTEM = """You are the evidence grader for a field-service troubleshooting assistant.
Decide which manual excerpts are directly useful for diagnosing or fixing the reported problem
on this equipment. An excerpt about different equipment, or that merely shares keywords, is NOT relevant.
Return JSON: {"relevant_ids": [chunk ids], "rationale": "one sentence"}."""

    def __init__(self, chat: ChatFn, min_score: float, escalation_score: float):
        self.chat = chat
        self.min_score = min_score
        self.escalation_score = escalation_score

    def run(self, question: str, triage: dict, evidence: list[Evidence]) -> tuple[list[Evidence], dict]:
        kept = [e for e in evidence if e.score >= self.min_score]
        top = max((e.score for e in evidence), default=0.0)
        detail = {"top_score": top, "above_min_score": len(kept)}
        if not kept or top < self.escalation_score:
            detail["escalate"] = (
                f"Best manual match scored {top:.2f}, below the escalation threshold "
                f"{self.escalation_score:.2f}; the manuals do not appear to cover this problem."
            )
            return [], detail

        user = (
            f"Problem: {question}\nTriage: {json.dumps(triage)}\n\nExcerpts:\n\n{_format_evidence(kept)}"
        )
        out = self.chat(self.SYSTEM, user)
        ids = set(out.get("relevant_ids", []))
        relevant = [e for e in kept if e.chunk_id in ids]
        detail.update(relevant_ids=[e.chunk_id for e in relevant], rationale=out.get("rationale", ""))
        if not relevant:
            detail["escalate"] = "Retrieved excerpts were judged not relevant to this problem."
        return relevant, detail


class DiagnosisAgent:
    SYSTEM = """You are the diagnosis agent for a field-service troubleshooting assistant.
Answer ONLY from the provided manual excerpts. Every cause, step and warning must cite the
excerpt ids it comes from. Never add procedures, values or part numbers that are not in the excerpts.
If the excerpts do not support a diagnosis, set "insufficient_evidence" to true.
Return JSON:
{
  "summary": "one or two sentences",
  "likely_causes": [{"cause": str, "citations": [ids]}],
  "steps": [{"instruction": str, "citations": [ids]}],
  "safety_warnings": [{"warning": str, "citations": [ids]}],
  "confidence": "high" | "medium" | "low",
  "insufficient_evidence": bool
}
Order steps from safest/simplest to most involved."""

    def __init__(self, chat: ChatFn):
        self.chat = chat

    def run(self, question: str, triage: dict, evidence: list[Evidence]) -> tuple[dict, dict]:
        user = (
            f"Problem: {question}\nTriage: {json.dumps(triage)}\n\nExcerpts:\n\n{_format_evidence(evidence)}"
        )
        out = self.chat(self.SYSTEM, user)
        valid_ids = {e.chunk_id for e in evidence}
        dropped = 0

        def grounded(items, text_key):
            nonlocal dropped
            kept = []
            for item in items or []:
                cites = [c for c in item.get("citations", []) if c in valid_ids]
                if item.get(text_key) and cites:
                    kept.append({text_key: item[text_key], "citations": cites})
                else:
                    dropped += 1
            return kept

        diagnosis = {
            "summary": out.get("summary", ""),
            "likely_causes": grounded(out.get("likely_causes"), "cause"),
            "steps": grounded(out.get("steps"), "instruction"),
            "safety_warnings": grounded(out.get("safety_warnings"), "warning"),
            "confidence": out.get("confidence", "low"),
        }
        detail = {"dropped_uncited_items": dropped}
        if out.get("insufficient_evidence") or not diagnosis["steps"]:
            detail["escalate"] = "The manual excerpts did not support a grounded diagnosis."
        return diagnosis, detail


class EscalationAgent:
    """Deterministic hand-off: no LLM, so it cannot invent unsupported advice."""

    def run(self, reason: str, triage: dict, evidence: list[Evidence]) -> dict:
        collect = ["Equipment make, model and serial number", "Exact error codes and when they appear",
                   "Photos of the rating plate and the fault area"]
        if not triage.get("error_codes"):
            collect.insert(1, "Any error code or indicator light pattern on the display")
        return {
            "reason": reason,
            "recommended_action": "Escalate to Tier-2 support / the manufacturer's technical line.",
            "information_to_collect": collect,
            "closest_manual_sections": [
                {"chunk_id": e.chunk_id, "source": e.source, "page": e.page, "section": e.section, "score": e.score}
                for e in evidence[:3]
            ],
        }


class Orchestrator:
    def __init__(
        self,
        store: VectorStore,
        embed: EmbedFn,
        chat: ChatFn,
        min_score: float = config.MIN_CHUNK_SCORE,
        escalation_score: float = config.ESCALATION_SCORE,
    ):
        self.triage = TriageAgent(chat)
        self.retrieval = RetrievalAgent(store, embed)
        self.grader = EvidenceGrader(chat, min_score, escalation_score)
        self.diagnosis = DiagnosisAgent(chat)
        self.escalation = EscalationAgent()

    def troubleshoot(self, question: str, equipment: str | None = None, top_k: int = config.TOP_K) -> Result:
        triage = self.triage.run(question, equipment)
        result = Result(status="resolved", question=question, triage=triage)
        result.trace.append(TraceStep("triage", triage))

        retrieved = self.retrieval.run(question, triage, top_k)
        result.trace.append(TraceStep("retrieval", {"hits": [(e.chunk_id, e.score) for e in retrieved]}))

        relevant, grade = self.grader.run(question, triage, retrieved)
        result.trace.append(TraceStep("evidence_grader", grade))
        if "escalate" in grade:
            return self._escalate(result, grade["escalate"], retrieved)

        diagnosis, diag_detail = self.diagnosis.run(question, triage, relevant)
        result.trace.append(TraceStep("diagnosis", diag_detail))
        result.evidence = relevant
        if "escalate" in diag_detail:
            return self._escalate(result, diag_detail["escalate"], relevant)

        if triage["safety_concerns"] and not diagnosis["safety_warnings"]:
            diagnosis["safety_warnings"].append({
                "warning": "Reported hazard (" + ", ".join(triage["safety_concerns"]) + "): make the "
                "site safe and follow your site safety procedure before servicing.",
                "citations": [],
            })
        result.diagnosis = diagnosis
        return result

    def _escalate(self, result: Result, reason: str, evidence: list[Evidence]) -> Result:
        result.status = "escalated"
        result.evidence = evidence
        result.escalation = self.escalation.run(reason, result.triage, evidence)
        result.trace.append(TraceStep("escalation", {"reason": reason}))
        return result
