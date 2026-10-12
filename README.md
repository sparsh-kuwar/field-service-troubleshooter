# Agentic Field-Service Troubleshooting Assistant

An AI-powered field-service troubleshooting assistant that uses Retrieval-Augmented Generation (RAG), FAISS semantic search, LLMs, and a lightweight multi-agent workflow to help technicians diagnose equipment problems from service manuals.

The system reduces unsupported AI answers by grounding recommendations in retrieved manual content and automatically escalating problems when the retrieved evidence is not relevant enough.

## How it works

```
technician report
      │
 1. Triage agent (LLM)        extracts equipment, symptoms, error codes, safety concerns, search queries
      │
 2. Retrieval agent           embeds the report + queries, FAISS cosine search over manual chunks
      │
 3. Evidence grader           score gate: best match < ESCALATION_SCORE  ──► escalate (no LLM answer)
      │                       LLM relevance check: no relevant excerpts  ──► escalate
 4. Diagnosis agent (LLM)     answers only from excerpts; every cause/step/warning must cite chunk ids.
      │                       Items citing nothing (or unknown ids) are dropped.
      │                       No grounded steps / "insufficient evidence"  ──► escalate
      ▼
 resolved (with citations)  or  escalated (deterministic hand-off: reason, info to collect, closest sections)
```

Every response includes a `trace` showing what each agent decided.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then put your OPENAI_API_KEY in .env
.venv/bin/uvicorn main:app --reload
```

## Usage

Put PDF, TXT or MD manuals in `manuals/` (a fictional sample manual is included), then:

```bash
curl -X POST localhost:8000/ingest                       # build the FAISS index
curl -X POST localhost:8000/troubleshoot -H 'Content-Type: application/json' \
  -d '{"question": "Showing E1, igniter clicks but no flame", "equipment": "ThermaFlow TF-200"}'
curl -F file=@my_manual.pdf localhost:8000/manuals       # upload a manual and re-index
```

Web UI: http://localhost:8000 · API docs: http://localhost:8000/docs

| Endpoint | Purpose |
|---|---|
| `GET /health` | Index status and models in use |
| `POST /ingest` | Rebuild the index from `manuals/` |
| `POST /manuals` | Upload a manual and rebuild |
| `POST /troubleshoot` | Run the agent workflow |

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_CHAT_MODEL` | `gpt-4.1-mini` | Model for triage, grading and diagnosis |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `MIN_CHUNK_SCORE` | `0.25` | Chunks below this cosine score are discarded |
| `ESCALATION_SCORE` | `0.35` | Escalate if the best chunk is below this |
| `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | `6`, `900`, `150` | Retrieval and chunking |

Tune the two score thresholds for your manuals: check `trace[evidence_grader].top_score` for known in-scope and out-of-scope questions.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests run offline, using a fake embedder and a scripted LLM.
