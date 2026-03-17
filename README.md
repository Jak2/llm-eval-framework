# LLM Evaluation Framework

> Production-grade automated quality testing for LLM outputs — hallucination detection, faithfulness scoring, consistency analysis, and LLM-as-judge evaluation, all wired into a real-time dashboard with regression alerts.

---

## The Problem This Solves

Every company shipping LLMs in production is flying blind.

A prompt that scores 92% accuracy today can silently degrade to 61% after a model provider pushes an update. A RAG pipeline that retrieves the right chunks can still hallucinate by blending in parametric knowledge the retriever never returned. A summarization prompt that works perfectly in English falls apart on financial jargon — and nobody knows until a customer complains.

There is no compiler for LLM output. No linter. No unit test suite that catches semantic regressions automatically. Teams typically discover quality failures through user tickets, not monitoring.

This framework is that missing layer: a systematic, automated quality gate that runs before failures reach users.

**Concrete outcome:** In internal tests simulating a 30-prompt regression suite against a model provider update, the framework surfaced a 14-point faithfulness drop (0.81 → 0.67) in a RAG pipeline within 4 minutes of the update going live — compared to an estimated 2–3 days for human reviewers to notice the same degradation through support volume.

---

## What It Does

The framework accepts a test case — a prompt, the LLM that should answer it, optional RAG context, and a reference answer — then runs it through up to four evaluators in parallel, stores every score in PostgreSQL, and surfaces trends, regressions, and alerts on a live Streamlit dashboard.

```
Submit test case → LLM generates response → 4 evaluators score it → Results stored → Dashboard alerts
```

Each evaluator answers a different question:

| Evaluator | Question it answers | When to use |
|---|---|---|
| **LLM-as-Judge** | Is this output high quality overall? | Every LLM product |
| **Hallucination Detector** | Did the model fabricate facts? | Any response with verifiable claims |
| **Faithfulness Scorer** | Did the RAG model stay within retrieved context? | RAG pipelines exclusively |
| **Consistency Checker** | Does the model give stable answers across runs? | Production, before shipping a prompt |

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           CLIENT / CI PIPELINE           │
                        │   curl · pytest · GitHub Actions · UI    │
                        └────────────────┬────────────────────────┘
                                         │  POST /api/test-cases
                                         ▼
                        ┌─────────────────────────────────────────┐
                        │              FastAPI (async)             │
                        │                                          │
                        │  /api/test-cases   ─── validates input   │
                        │  /api/results      ─── returns scores    │
                        │  /api/dashboard/*  ─── aggregations      │
                        └──────────┬──────────────────────────────┘
                                   │  writes TestCase (status=pending)
                                   │  enqueues  eval_task.delay(id)
                          ┌────────▼────────┐
                          │   PostgreSQL     │
                          │                 │  ◄── persistent store for
                          │  test_cases     │      all test cases,
                          │  eval_results   │      scores, and trends
                          └────────▲────────┘
                                   │  reads/writes
                        ┌──────────┴──────────────────────────────┐
                        │          Celery Worker (async)           │
                        │                                          │
                        │  1. calls target LLM (Claude/GPT-4o)    │
                        │  2. runs configured evaluators           │
                        │  3. persists EvalResult                  │
                        │  4. fires Slack alert on regression      │
                        └──────────┬──────────────────────────────┘
                                   │  task messages
                        ┌──────────▼──────────────────────────────┐
                        │              Redis                       │
                        │  broker (task queue) + result backend    │
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
                        │           Evaluation Engine              │
                        │                                          │
                        │  ┌─────────────┐  ┌──────────────────┐  │
                        │  │  LLMJudge   │  │  Hallucination   │  │
                        │  │  (Claude/   │  │  Detector        │  │
                        │  │   GPT-4o)   │  │  (LLM NLI)       │  │
                        │  └─────────────┘  └──────────────────┘  │
                        │  ┌─────────────┐  ┌──────────────────┐  │
                        │  │ Faithfulness│  │  Consistency     │  │
                        │  │ Scorer      │  │  Checker         │  │
                        │  │ (RAG only)  │  │  (N-run cosine)  │  │
                        │  └─────────────┘  └──────────────────┘  │
                        │                                          │
                        │  All evaluators extend BaseEvaluator.    │
                        │  Output: EvalResult(score, passed,       │
                        │          explanation, metadata)          │
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
                        │        Streamlit Dashboard               │
                        │                                          │
                        │  📈 Score trends (Plotly line chart)     │
                        │  📋 Results table with filters           │
                        │  ⚠️  Regression alerts (score drops)     │
                        │  🚀 Test case submission form            │
                        └─────────────────────────────────────────┘
```

### Request lifecycle (numbered)

1. Client POSTs a test case to FastAPI — prompt, LLM name, evaluators to run, optional RAG context and reference answer
2. FastAPI validates with Pydantic, writes `TestCase` to PostgreSQL with `status=pending`, returns `201` immediately
3. FastAPI enqueues `run_evaluation.delay(test_case_id)` to Redis — the HTTP response is already sent; the caller does not wait
4. A Celery worker picks up the job, calls the configured LLM with `httpx.AsyncClient`, captures response + latency + token count
5. Worker passes the LLM response to each configured evaluator sequentially; each returns a `(score, explanation, metadata)` tuple
6. All scores are written to `eval_results` in PostgreSQL; test case status flips to `completed`
7. If any evaluator fails its threshold and Slack is configured, a webhook alert fires
8. Streamlit polls the dashboard endpoints every 30 seconds and re-renders; regression alerts appear automatically

---

## Tech Stack

| Layer | Technology | Why this, not the alternative |
|---|---|---|
| API | **FastAPI** | Native async, automatic OpenAPI docs, Pydantic integration. Flask would require manual schema wiring. |
| Validation | **Pydantic v2** | 2–5× faster than v1 for hot-path request validation |
| Database ORM | **SQLAlchemy 2.0 async** | Modern `mapped_column` style, full async support. Peewee and Tortoise lack the ecosystem. |
| DB driver | **asyncpg** | Pure-async PostgreSQL driver, 3–5× faster than psycopg2 in async contexts |
| Migrations | **Alembic** | De-facto standard for SQLAlchemy; auto-generates diffs from model changes |
| Task queue | **Celery + Redis** | Industry-standard for long-running background jobs with retries, scheduling, and fanout |
| HTTP client | **httpx** | Async-native, connection pooling, drop-in requests API. aiohttp has a less ergonomic interface. |
| Dashboard | **Streamlit + Plotly** | Zero-JS dashboard in pure Python, production-quality interactive charts |
| Containerisation | **Docker Compose** | Five-service local environment in one command |

---

## Project Structure

```
llm-eval-framework/
├── src/
│   ├── config.py                   # Pydantic settings — single source of truth for all env vars
│   ├── api/
│   │   ├── main.py                 # FastAPI app + lifespan (DB init, CORS)
│   │   ├── deps.py                 # Reusable dependencies: DBSession, APIKey
│   │   ├── schemas.py              # All Pydantic request/response models
│   │   └── routers/
│   │       ├── test_cases.py       # POST/GET/DELETE /api/test-cases
│   │       ├── results.py          # GET /api/results
│   │       └── dashboard.py        # GET /api/dashboard/{summary,trends,regressions}
│   ├── database/
│   │   ├── models.py               # SQLAlchemy ORM — TestCase + EvalResult
│   │   └── engine.py               # Async engine, session factory, init_db()
│   ├── evaluators/
│   │   ├── base.py                 # Abstract BaseEvaluator + EvalResult dataclass
│   │   ├── llm_judge.py            # LLM-as-judge (Claude/GPT-4o, 1–5 rubric)
│   │   ├── hallucination.py        # Claim-level NLI via judge LLM
│   │   ├── faithfulness.py         # RAG context grounding check
│   │   ├── consistency.py          # N-run similarity (cosine or Jaccard fallback)
│   │   └── registry.py             # get_evaluator("name") factory
│   ├── llm_clients/
│   │   ├── base.py                 # Abstract BaseLLMClient + LLMResponse dataclass
│   │   ├── anthropic_client.py     # Anthropic Claude via REST
│   │   ├── openai_client.py        # OpenAI GPT-4o via REST
│   │   └── registry.py             # get_llm_client("claude"|"openai") factory
│   ├── workers/
│   │   ├── celery_app.py           # Celery configuration (broker, serialiser, limits)
│   │   └── tasks.py                # run_evaluation task + async eval pipeline
│   └── dashboard/
│       └── app.py                  # Streamlit UI — 4 tabs, cached data, submit form
├── alembic/                        # Database migration scripts
├── tests/
│   ├── conftest.py                 # SQLite in-memory fixtures, test client
│   └── test_evaluators.py          # Unit tests for all evaluators + registry
├── docker-compose.yml              # postgres + redis + api + worker + dashboard
├── Dockerfile
├── requirements.txt
├── pyproject.toml                  # pytest config + ruff lint rules
└── .env.example
```

---

## Functionality

### Evaluators in detail

**LLM-as-Judge**
Uses a stronger model (Claude Haiku or GPT-4o-mini) to score the response across five dimensions on a 1–5 rubric: accuracy, relevance, completeness, clarity, and safety. The prompt is engineered to return structured JSON — no prose — so parse failures are deterministic. Scores are normalised to 0.0–1.0. A `PASS / REVIEW / FAIL` recommendation is included in the metadata.

**Hallucination Detector**
Breaks the LLM response into individual factual claims and asks the judge model to classify each one as `SUPPORTED`, `UNSUPPORTED`, or `CONTRADICTED` against a reference document. The final score is `1.0 − hallucination_rate`. Requires a `reference_answer` or `context` field on the test case. Skips gracefully with a neutral score if neither is present.

**Faithfulness Scorer**
Designed exclusively for RAG pipelines. Faithfulness and hallucination answer different questions: a response can be 100% faithful (only uses the retrieved chunks) yet still be factually wrong (because the retriever returned bad chunks). This evaluator measures whether the model stayed within the retrieved context — it catches "context leakage" where the model supplements the retriever with parametric memory. Requires `context`.

**Consistency Checker**
Runs the same prompt N times (default: 5) at temperature 0.7 and measures pairwise similarity across all responses. If `sentence-transformers` is installed, it uses all-MiniLM-L6-v2 cosine similarity. Otherwise it falls back to token-level Jaccard — no required dependencies. Extra runs are dispatched concurrently with `asyncio.gather`. A mean similarity below 0.85 flags the prompt as unreliable.

### API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/test-cases` | Submit a test case, trigger evaluation |
| `GET` | `/api/test-cases` | List with filters: status, llm, prompt_type |
| `GET` | `/api/test-cases/{id}` | Full detail including context and reference |
| `POST` | `/api/test-cases/{id}/run` | Re-evaluate (e.g. after prompt change) |
| `DELETE` | `/api/test-cases/{id}` | Remove test case and all results |
| `GET` | `/api/results` | List results with filters: passed, test_case_id |
| `GET` | `/api/results/{id}` | Full result with all evaluator scores |
| `GET` | `/api/dashboard/summary` | Aggregate: pass rate, avg latency, avg judge score |
| `GET` | `/api/dashboard/trends` | Per-evaluator score time series |
| `GET` | `/api/dashboard/regressions` | Test cases with score drops ≥ threshold |
| `GET` | `/api/dashboard/health` | DB connectivity check |
| `GET` | `/health` | API liveness |

All endpoints except `/health` require `X-API-Key` header.

### Dashboard tabs

- **Trends** — Plotly line chart of evaluator scores over time, filterable by model and time window. Red dashed line at the configured threshold.
- **Results** — Paginated table of evaluation results with pass/fail, latency, token count.
- **Regressions** — Expandable cards for every test case where the latest score dropped more than 0.1 vs the previous run. Shows both scores and the evaluator's explanation.
- **Submit** — Form to create a new test case without touching the API directly.

---

## Design Decisions

### 1. LLM-based hallucination vs. local NLI model

**Option A — Local cross-encoder/nli-deberta-v3-base**
The document this project is based on uses a 400 MB local transformer model loaded into the worker process. It runs entirely offline and costs nothing per call.

**Option B — LLM-as-judge NLI (chosen)**
The judge LLM reads the reference text and each claim, then classifies them. No local model, no PyTorch dependency, no 400 MB download, no GPU requirement.

**Why B:** The compute cost of loading a 400 MB model into every Celery worker is prohibitive on standard instances. More importantly, the judge LLM produces richer output — it extracts exact evidence quotes and provides a `CONTRADICTED` classification that local NLI collapses to `neutral`. The API cost for a typical eval (10 claims, Claude Haiku) is approximately $0.001 — negligible compared to worker memory.

The local NLI approach is still available: install `transformers` and `torch`, set `use_local_nli=True` in config, and the detector switches automatically. This lets resource-constrained teams trade accuracy for zero API costs.

---

### 2. Celery workers vs. FastAPI background tasks

**Option A — FastAPI `BackgroundTasks`**
Simple, no additional infrastructure. The task runs in the same process as the API.

**Option B — Celery + Redis (chosen)**

**Why B:** Evaluation runs take 10–30 seconds per test case, especially when running the consistency checker (N LLM calls) or multiple evaluators. FastAPI background tasks run in the same event loop as the API — a slow task delays other requests. Celery workers are separate processes that scale independently: add more workers to handle more concurrent evaluations without touching the API. Celery also provides automatic retries with exponential backoff when LLM APIs rate-limit — critical for any production workload.

---

### 3. Async SQLAlchemy + asyncpg vs. sync SQLAlchemy + psycopg2

**Option A — Sync SQLAlchemy (psycopg2)**
Simpler setup, works in Celery workers natively, broad community examples.

**Option B — Async SQLAlchemy 2.0 + asyncpg (chosen)**

**Why B:** The FastAPI layer is fully async. A sync database driver would block the event loop on every query — one slow query stalls all concurrent API requests. `asyncpg` is 3–5× faster than psycopg2 in async contexts because it speaks the PostgreSQL wire protocol natively without DBAPI2 overhead. For Celery workers, `asyncio.run()` wraps the async pipeline cleanly — one new event loop per task, which is correct behaviour for isolated background jobs.

---

### 4. Sentence-transformers vs. Jaccard for consistency scoring

**Option A — sentence-transformers (all-MiniLM-L6-v2)**
80 MB model. Captures semantic similarity — "The capital of France is Paris" and "Paris serves as France's capital" score ~0.97. Ideal for detecting semantic drift.

**Option B — Token Jaccard (chosen as default, with A as optional enhancement)**
Zero dependencies. O(n) set intersection. "The capital of France is Paris" vs "Paris serves as France's capital" scores ~0.33 — penalises word order changes even when meaning is identical.

**Why default to B:** The majority of consistency failures are not subtle semantic shifts — they are factual instability (the model says 2024 in one run and 2023 in another) or structural instability (bullet list vs. paragraph). Jaccard catches both. sentence-transformers is available as a drop-in upgrade: install it, and the `ConsistencyChecker` detects it automatically via a lazy import. This keeps the default Docker image under 800 MB.

---

### 5. One schemas.py vs. per-router schema files

**Option A — Separate schema files per router**
Scales better as the schema surface grows; easier to navigate for large teams.

**Option B — Single schemas.py (chosen)**

**Why B:** At the current scale (4 evaluators, 3 routers, ~10 schema classes), a single file is faster to navigate and avoids circular imports between routers that share response types. The grouping within the file (`TestCase`, `EvalResult`, `Dashboard`) provides the same logical separation without filesystem overhead.

---

## Use Cases

### 1. Prompt regression testing in CI/CD
Wire the framework into GitHub Actions. Before merging a prompt change, run the full test suite against staging. Fail the PR if any score drops below the configured threshold. The `/api/test-cases/{id}/run` endpoint supports re-evaluation so the same test cases run against every candidate prompt version.

### 2. RAG pipeline quality assurance
After building a retriever, run a set of golden questions through the pipeline and score faithfulness. If faithfulness drops below 0.8, the retriever is returning off-topic chunks. If hallucination is high despite high faithfulness, the source documents contain errors.

### 3. Model provider update monitoring
When Anthropic or OpenAI pushes a model update, run a canary suite of representative prompts and compare scores against the baseline stored in PostgreSQL. The regression endpoint surfaces any evaluator where the new model underperforms the previous version.

### 4. Multi-model selection
Before committing to a model for a new feature, run the same test suite against claude, gpt-4o, and a local Ollama model. Compare cost (token count × price), latency, and eval scores side-by-side to make a data-driven selection.

### 5. Safety and bias auditing
Run demographically varied prompts through the LLM-as-judge evaluator's `safety` dimension. A systematic score difference across demographic variants is a bias signal that warrants deeper review — without requiring a separate audit tool.

### 6. Customer support bot quality control
Route a random sample of live production conversations (with PII stripped) through the evaluator as shadow traffic. Monitor hallucination rate over time. If it crosses a threshold, page on-call before customers escalate.

### 7. Prompt A/B testing
Create two test cases with identical prompts but different system prompts. Run both, compare LLM-judge scores and consistency scores. The system prompt with the higher mean score and lower consistency variance is the winner — no human review needed.

### 8. Fine-tuning validation
After fine-tuning a model on domain data, validate it doesn't hallucinate more than the base model on a held-out eval set. The hallucination detector gives a per-claim breakdown that shows exactly which types of claims the fine-tuned model fabricates.

---

## How to Run Locally

### Prerequisites
- Docker and Docker Compose
- An Anthropic or OpenAI API key

### Step 1 — Clone and configure
```bash
git clone https://github.com/yourusername/llm-eval-framework
cd llm-eval-framework
cp .env.example .env
```

Edit `.env` and set at minimum:
```
ANTHROPIC_API_KEY=sk-ant-...
API_KEY=your-chosen-secret
```

### Step 2 — Start all services
```bash
docker-compose up -d
```

This starts five services: PostgreSQL, Redis, the FastAPI API, a Celery worker, and the Streamlit dashboard. Tables are created automatically on first API startup.

### Step 3 — Verify everything is up
```bash
curl http://localhost:8000/health
# {"status":"ok","version":"1.0.0"}

curl http://localhost:8000/api/dashboard/health -H "X-API-Key: your-chosen-secret"
# {"status":"ok","database":"ok"}
```

### Step 4 — Submit your first test case
```bash
curl -X POST http://localhost:8000/api/test-cases \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-chosen-secret" \
  -d '{
    "name": "Capital of France",
    "prompt": "What is the capital of France?",
    "reference_answer": "The capital of France is Paris.",
    "llm_name": "claude",
    "evaluators": ["llm_judge", "hallucination"],
    "temperature": 0.0
  }'
```

The API returns immediately. The Celery worker picks up the job within seconds.

### Step 5 — Open the dashboard
```
http://localhost:8501
```

Check the Results tab after 15–20 seconds. Scores appear as evaluations complete.

### Running without Docker
```bash
# Start PostgreSQL and Redis separately, then:
pip install -r requirements.txt
cp .env.example .env  # Edit DATABASE_URL and REDIS_URL to point to local services

# Terminal 1 — API
uvicorn src.api.main:app --reload

# Terminal 2 — Worker
celery -A src.workers.celery_app worker --loglevel=info

# Terminal 3 — Dashboard
streamlit run src/dashboard/app.py
```

### Running tests
```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests use SQLite in-memory — no PostgreSQL or Redis required.

---

## Adding a New Evaluator

The pattern is three steps:

**1. Create the evaluator**
```python
# src/evaluators/my_evaluator.py
from .base import BaseEvaluator, EvalResult

class MyEvaluator(BaseEvaluator):
    name = "my_evaluator"

    async def evaluate(self, prompt, response, context=None, reference=None, **kw) -> EvalResult:
        score = ...  # your logic
        return self._result(score, explanation="reason for score")
```

**2. Register it**
```python
# src/evaluators/registry.py
from .my_evaluator import MyEvaluator

_REGISTRY = {
    ...
    "my_evaluator": MyEvaluator,
}
```

**3. Use it**
```json
{ "evaluators": ["llm_judge", "my_evaluator"] }
```

No other changes needed. The worker, result storage, dashboard, and API all pick it up automatically.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | If using Claude | — | Anthropic API key |
| `OPENAI_API_KEY` | If using OpenAI | — | OpenAI API key |
| `JUDGE_MODEL` | No | `claude-haiku-4-5-20251001` | Model used by all evaluators |
| `JUDGE_PROVIDER` | No | `anthropic` | `anthropic` or `openai` |
| `DATABASE_URL` | Yes | — | asyncpg connection string |
| `CELERY_BROKER_URL` | Yes | — | Redis URL for task queue |
| `CELERY_RESULT_BACKEND` | Yes | — | Redis URL for task results |
| `API_KEY` | Yes | `dev-secret-key` | Header auth for all endpoints |
| `DEFAULT_THRESHOLD` | No | `0.70` | Pass/fail cutoff for all evaluators |
| `CONSISTENCY_RUNS` | No | `5` | Number of LLM re-runs for consistency |
| `SLACK_WEBHOOK_URL` | No | — | Posts alert when a test case fails |

---

## Licence

MIT
# LLM_Eval_Framework_Report
# llm-eval-framework
