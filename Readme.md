# Production-Grade Enterprise Document Intelligence & RAG Platform

An enterprise-grade **Knowledge Intelligence Platform**: ingest PDF / DOCX /
Markdown / TXT / web documents, ask natural-language questions, and receive
**grounded answers with citations, sources, and confidence estimates**.

This is a RAG *platform* — hybrid retrieval (dense + sparse + fusion), cross-encoder
reranking, grounding verification, document versioning, an evaluation harness, and
production observability. It is not a toy chatbot.

> **Status:** Phase 2 of 34 complete (repository & development foundation).
> Progress: see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Quickstart (Docker)

```bash
cp .env.example .env
docker compose up -d --build
bash scripts/smoke_test.sh