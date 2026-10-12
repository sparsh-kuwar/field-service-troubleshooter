# Agentic Field-Service Troubleshooting Assistant

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/sparsh-kuwar/field-service-troubleshooter)

An AI-powered troubleshooting assistant for field-service technicians. It combines **Retrieval-Augmented Generation (RAG)**, **FAISS semantic search**, **OpenAI LLMs** and a **multi-agent workflow** to diagnose equipment faults from service manuals. Every recommendation **cites the exact manual section it came from**.

The system is built so that it does not guess. It escalates to a human whenever the manuals do not clearly support an answer, and it removes any AI-generated step that cannot be traced back to the manual.

It ships as a **FastAPI service with a web UI**, offline tests, a Dockerfile and a one-click Render deployment.

---

## 🚀 Project Overview

Field technicians often troubleshoot equipment on site with only a short description of the fault and a long PDF manual. Searching that manual by hand is slow. A plain chatbot is risky because it can invent procedures, torque values or part numbers that are not in the manual.

This project takes a technician's free-text report and:

1. **Triages** it into equipment, symptoms, error codes, safety hazards and manual-style search queries.
2. **Retrieves** the most relevant manual sections using FAISS cosine-similarity search.
3. **Grades the evidence** with a numeric score gate followed by an LLM relevance check.
4. **Diagnoses** the fault using only the retrieved excerpts. Every cause, step and warning must cite a chunk id.
5. **Verifies grounding.** Any item that cites nothing, or cites an unknown chunk, is dropped.
6. **Escalates** with a deterministic, non-LLM hand-off whenever the evidence is weak.
7. Returns a full **agent trace** that shows what each agent decided and why.

---

## 🏗️ Architecture

```text
                    Service manuals (PDF / TXT / MD)
                                  |
                                  v
                Page loader (pypdf) + heading-aware chunker
                                  |
                                  v
              OpenAI embeddings (text-embedding-3-small)
                                  |
                                  v
            FAISS IndexFlatIP over L2-normalised vectors
                     (cosine similarity, persisted)
                                  |
     ============================ | ============================
                                  |
                   Technician report (+ equipment)
                                  |
                                  v
                    +----------------------------+
                    | 1. Triage Agent (LLM)      |  equipment, symptoms, error codes,
                    +----------------------------+  safety concerns, search queries
                                  |
                                  v
                    +----------------------------+
                    | 2. Retrieval Agent         |  multi-query FAISS search,
                    +----------------------------+  best score per chunk
                                  |
                                  v
                    +----------------------------+
                    | 3. Evidence Grader         |
                    |   a) score gate            |-- top score < 0.35 --------+
                    |   b) LLM relevance check   |-- nothing relevant --------+
                    +----------------------------+                            |
                                  |                                           |
                                  v                                           |
                    +----------------------------+                            |
                    | 4. Diagnosis Agent (LLM)   |                            |
                    |   cite-or-drop filter      |-- no grounded steps -------+
                    +----------------------------+                            |
                                  |                                           v
                                  v                         +----------------------------+
                  RESOLVED: causes, steps, safety           | 5. Escalation Agent        |
                  warnings, each with citations             |    (deterministic, no LLM) |
                                                            +----------------------------+
                                                                          |
                                                                          v
                                                   ESCALATED: reason, info to collect,
                                                   closest manual sections
```

---

## 🧠 Core Technologies

| Technology | Purpose |
|---|---|
| Python 3.11+ | Application language |
| FastAPI + Uvicorn | REST API and web server |
| OpenAI API (`gpt-4.1-mini`) | Triage, evidence grading and diagnosis in JSON mode |
| OpenAI Embeddings (`text-embedding-3-small`) | Semantic vectors for manual chunks and queries |
| FAISS (`faiss-cpu`) | Vector similarity search |
| NumPy | Vector handling |
| pypdf | PDF text extraction, page by page |
| Pydantic | Request validation |
| python-dotenv | Environment configuration |
| pytest | Offline test suite |
| Docker + Render | Containerised one-click deployment |

There is no LangChain. The pipeline is written in plain Python, so every step can be inspected and every dependency can be swapped out in tests.

---

## 🔍 How the System Works

### 1. Manual ingestion and chunking (`app/ingest.py`)

Manuals go in `manuals/`. PDFs are read page by page, so citations can include page numbers. TXT and Markdown files are read whole.

The chunker is **heading-aware**. It recognises Markdown headings, numbered headings (`2.1 Error Codes`) and ALL-CAPS headings, and it starts a new chunk at each section boundary, so one error code's procedure does not mix with the next one.

- Paragraphs are packed up to `CHUNK_SIZE` (900) characters.
- Overly long paragraphs are split on sentence boundaries.
- `CHUNK_OVERLAP` (150) characters are carried into the next chunk so that context at a boundary is not lost.
- Each chunk records its `source`, `page` and `section`. The section heading is prepended before embedding, so short chunks still carry their context.

### 2. Vector store (`app/vectorstore.py`)

Embeddings are L2-normalised and stored in a FAISS `IndexFlatIP`. With normalised vectors, the inner product equals **cosine similarity**, so scores fall between −1 and 1 and are easy to interpret as thresholds. The index and chunk metadata are saved to `index/` and reloaded on startup.

The retriever runs **several queries at once** (the original report, the triage agent's search queries and any error codes) and keeps each chunk's best score.

### 3. Triage Agent

This agent turns a messy report such as *"Showing E1, igniter clicks but no flame"* into structured facts. It deliberately does **not** diagnose. It also rewrites the problem the way a service manual would phrase it, which noticeably improves retrieval for informal technician language.

### 4. Evidence Grader: two-stage confidence gate

- **Score gate (no LLM):** chunks below `MIN_CHUNK_SCORE` (0.25) are discarded. If the best chunk is below `ESCALATION_SCORE` (0.35), the request is escalated immediately and the LLM is never asked for an answer.
- **LLM relevance check:** the remaining excerpts are judged for real relevance. An excerpt about different equipment, or one that only shares keywords, is rejected.

### 5. Diagnosis Agent with cite-or-drop grounding

The LLM must return likely causes, ordered steps (safest first) and safety warnings, each with `citations` to chunk ids. The code then **checks the citations**:

- items with no citations, or with citations to chunks that were not provided, are **dropped** and counted in the trace;
- if no grounded steps remain, or the model reports `insufficient_evidence`, the request is **escalated**.

If triage detected a hazard (for example, a gas smell) but the diagnosis has no matching warning, a safety warning is added.

### 6. Escalation Agent (deterministic)

Escalation deliberately does **not** use an LLM, so it cannot invent advice. It returns the reason, a recommended action (Tier-2 or manufacturer support), a checklist of information to collect (model, serial number, error codes, photos) and the closest manual sections it found.

---

## 🧪 Example

Using the bundled (fictional) **ThermaFlow TF-200** tankless water heater manual:

**Request**

```bash
curl -X POST localhost:8000/troubleshoot -H 'Content-Type: application/json' \
  -d '{"question": "Showing E1, igniter clicks but no flame", "equipment": "ThermaFlow TF-200"}'
```

**Response (abridged)**

```json
{
  "status": "resolved",
  "diagnosis": {
    "summary": "E1 is an ignition failure: three ignition trials without detecting flame.",
    "likely_causes": [
      {"cause": "Manual gas valve closed or air in a new gas line", "citations": ["c3"]},
      {"cause": "Inlet gas pressure below 4.0 in. W.C. (natural gas)", "citations": ["c3"]},
      {"cause": "Fouled/cracked igniter or incorrect electrode gap", "citations": ["c3"]}
    ],
    "steps": [
      {"instruction": "Confirm the manual gas valve is fully open.", "citations": ["c3"]},
      {"instruction": "Measure inlet gas pressure; natural gas must stay between 4.0 and 10.5 in. W.C.", "citations": ["c3"]},
      {"instruction": "Check igniter gap is 0.12–0.16 in.; replace igniter TF-IGN-01 if cracked or eroded.", "citations": ["c3"]},
      {"instruction": "Reset the unit by pressing ON/OFF twice.", "citations": ["c3"]}
    ],
    "safety_warnings": [
      {"warning": "Disconnect 120 VAC power before removing the front cover; the igniter produces high voltage.", "citations": ["c1"]}
    ],
    "confidence": "high"
  },
  "evidence": [ { "chunk_id": "c3", "section": "E1 — Ignition Failure", "score": 0.62, "...": "..." } ],
  "trace": [ "triage", "retrieval", "evidence_grader", "diagnosis" ]
}
```

**Out-of-scope request:** *"Forklift hydraulic lift drifts down under load"*

```text
status: escalated
reason: Best manual match scored 0.18, below the escalation threshold 0.35;
        the manuals do not appear to cover this problem.
```

(Chunk ids and scores depend on the embedding model; the values above are illustrative.)

---

## ⚡ Quick Start (local)

```bash
git clone https://github.com/sparsh-kuwar/field-service-troubleshooter.git
cd field-service-troubleshooter
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env            # add your OPENAI_API_KEY
.venv/bin/uvicorn main:app --reload
```

Open **http://localhost:8000** for the web UI or **http://localhost:8000/docs** for interactive API docs. On the first start, the index is built automatically from `manuals/`.

### With Docker

```bash
docker build -t fst .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... fst
```

---

## ☁️ Deploy (one click)

1. Click **Deploy to Render** at the top of this README. A free Render account is enough.
2. Paste your `OPENAI_API_KEY` when prompted.
3. Render builds the Docker image, starts the service and indexes the sample manual on boot.

Render also generates a random `ADMIN_TOKEN`. While it is set, visitors can ask questions, but **uploading manuals and re-indexing are locked**, so strangers cannot use your OpenAI credits to embed files. To upload as the owner, pass the token in a header:

```bash
curl -F file=@my_manual.pdf -H "X-Admin-Token: <token>" https://<your-app>.onrender.com/manuals
```

The same Dockerfile also runs on Railway, Fly.io or Google Cloud Run. The container listens on `$PORT`.

### ⚠️ Uploaded manuals are temporary on the free tier

Render's free instances have an **ephemeral disk** that is wiped on every restart, redeploy or sleep/wake cycle.

- **Bundled manuals are safe.** Anything committed to `manuals/` ships inside the image. If no index exists at startup, the app re-indexes these manuals automatically before serving requests, so the demo always works.
- **Uploaded manuals are not kept.** Files uploaded through the UI or `POST /manuals` live only until the next restart and then disappear. To make a manual permanent, commit it to `manuals/` and redeploy, or attach a persistent disk (paid plan) and point `MANUALS_DIR` / `INDEX_DIR` at it.
- **The first request after a sleep is slow.** It takes about 30–60 s while the instance wakes up and rebuilds the index.

---

## 🔌 API

| Endpoint | Purpose |
|---|---|
| `GET /` | Web UI |
| `GET /health` | Index status, models in use, whether admin endpoints are locked |
| `POST /troubleshoot` | Run the agent workflow: `{"question": str, "equipment": str?, "top_k": int?}` |
| `POST /ingest` | Rebuild the index from `manuals/` *(admin)* |
| `POST /manuals` | Upload a PDF/TXT/MD manual and rebuild *(admin)* |
| `GET /docs` | OpenAPI / Swagger UI |

---

## ⚙️ Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_CHAT_MODEL` | `gpt-4.1-mini` | Model for triage, grading and diagnosis |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `MIN_CHUNK_SCORE` | `0.25` | Chunks below this cosine score are discarded |
| `ESCALATION_SCORE` | `0.35` | Escalate if the best chunk is below this |
| `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | `6`, `900`, `150` | Retrieval and chunking |
| `ADMIN_TOKEN` | unset | When set, `/ingest` and `/manuals` require `X-Admin-Token` |
| `MANUALS_DIR`, `INDEX_DIR` | `manuals/`, `index/` | Storage locations |

**Tuning the thresholds:** run known in-scope and out-of-scope questions and compare `trace[evidence_grader].top_score`. Then set `ESCALATION_SCORE` between the two groups.

---

## 🧪 Testing

```bash
.venv/bin/python -m pytest -q
```

The tests run **fully offline**. A hashed bag-of-words embedder and a scripted LLM replace OpenAI, which is possible because the agents receive `embed` and `chat` as injected functions. The tests cover:

- the chunker respecting size limits and tracking section headings,
- an in-scope fault resolving with only properly cited items kept,
- a low-similarity (out-of-scope) query being escalated by the score gate,
- a diagnosis flagged `insufficient_evidence` being escalated,
- the admin token locking upload and re-index endpoints.

---

## 🔒 Safety and Grounding

| Safeguard | How it is enforced |
|---|---|
| Manual-grounded answers | The diagnosis prompt only sees retrieved excerpts |
| No unsupported claims | **Code-level** citation check drops anything uncited |
| Confidence gate | Numeric cosine threshold, followed by an LLM relevance check |
| Human-in-the-loop | Weak evidence leads to escalation, never to a guess |
| No hallucinated hand-offs | The escalation path is deterministic and does not use an LLM |
| Hazard awareness | Safety concerns found in triage always produce a warning |
| Transparency | Every response includes a per-agent `trace` |
| Safe public demo | `ADMIN_TOKEN` locks costly or mutating endpoints |

---

## 🚧 Current Limitations

- Thresholds are tuned by hand; there is no labelled evaluation set yet.
- One sample manual is bundled, and it is fictional.
- The LLM calls (triage → grade → diagnose) run in sequence, which adds latency and cost.
- The index is a flat FAISS index that is rebuilt in full; there is no incremental indexing.
- There are no user accounts, no history or persistent storage beyond the local index.
- Scanned (image-only) PDFs need OCR before ingestion.

## 🔮 Future Improvements

1. An evaluation harness measuring retrieval recall and grounded-answer rate.
2. Metadata filtering by equipment model across many manuals.
3. Incremental indexing and a managed vector DB (pgvector, Qdrant).
4. Streaming responses and parallel agent calls.
5. Ticket history and technician feedback loops.
6. OCR for scanned manuals and image input for fault photos.
7. Observability (LangSmith / OpenTelemetry) and CI/CD.

---

## 📁 Project Structure

```text
.
├── main.py               FastAPI app: routes, startup indexing, admin guard
├── app/
│   ├── config.py         Environment-driven settings
│   ├── ingest.py         Loading, heading-aware chunking, index build
│   ├── vectorstore.py    FAISS cosine index + persistence
│   ├── agents.py         Triage, Retrieval, Grader, Diagnosis, Escalation, Orchestrator
│   └── llm.py            Thin OpenAI wrapper (embed, chat_json)
├── static/index.html     Single-page web UI with clickable citations
├── manuals/              Service manuals to index (sample included)
├── tests/                Offline pytest suite
├── Dockerfile            Container image
└── render.yaml           One-click Render blueprint
```

---

## 💡 Key Concepts Demonstrated

Retrieval-Augmented Generation · semantic search · embeddings · FAISS · structure-aware chunking · multi-query retrieval · multi-agent orchestration · prompt engineering with JSON outputs · citation-based grounding verification · confidence-gated escalation · human-in-the-loop design · dependency injection for testable LLM code · FastAPI · Docker · cloud deployment

---

## 👨‍💻 Author

**Sparsh Kuwar**
Computer Science Engineering, JSPM Rajarshi Shahu College of Engineering
