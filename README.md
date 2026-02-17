# SevAI / ProdLens

ProdLens is a **decision-oriented research agent** that answers questions like:

- “Should we use RAG or fine-tuning for our customer support bot?”
- “Should we move from OpenAI to an open‑source model for this workload?”

Instead of just explaining the tech, ProdLens produces:

- **Evidence from papers, blogs, and GitHub issues**
- **Tradeoff matrices** across latency, cost, quality, complexity, and risk
- **Rough cost and latency estimates**
- A **production recommendation** with a **readiness score (0–100)**

It supports:

- **Quick Mode** – 5‑bullet, high‑signal recommendation
- **Deep Mode** – structured decision memo with evidence and scores

This repo contains a small FastAPI service + CLI that implement ProdLens.

## Getting Started

### 1. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows PowerShell
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Set the following environment variables as needed:

- `PRODLENS_OPENAI_API_KEY` – API key for your LLM provider (e.g. OpenAI-compatible endpoint).
- `PRODLENS_QDRANT_URL` – Qdrant endpoint (e.g. `http://localhost:6333`).
- `PRODLENS_QDRANT_API_KEY` – Qdrant API key if authentication is enabled (optional).

You can run Qdrant locally via Docker, for example:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Run the API server

```bash
uvicorn app.main:app --reload
```

The FastAPI docs will be available at `http://localhost:8000/docs`.

### 5. Use the CLI

Quick mode:

```bash
python cli.py quick "RAG vs fine-tuning for customer support; optimize for latency over cost"
```

Deep mode:

```bash
python cli.py deep "Should we use RAG or fine-tuning for our customer support bot? Latency over cost, data is proprietary."
```

Both CLI and API will:

- Use **Qdrant** to persist prior research notes and user preference profiles.
- Call the configured LLM to synthesize:
  - Evidence + citations
  - Tradeoff matrix
  - Cost / latency estimates
  - A production recommendation and readiness score.
