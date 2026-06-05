# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG chatbot that answers questions about hikes in Washington state (difficulty, distance, elevation, best season, permits, trail conditions). Built with LangChain + Chroma for retrieval and Google ADK for the agent.

## Architecture

Two decoupled components:

1. **`data_ingestion.ipynb`** — one-time pipeline that scrapes web sources, chunks documents, embeds them with `text-embedding-3-small`, and persists to a local Chroma vector store at `./chroma_db` (collection: `washington_hikes`)

2. **`hiking_agent/agent.py`** — Google ADK agent (`gemini-2.5-flash`) with a single `retrieve_info` tool that queries the Chroma store at runtime to answer hiking questions

The agent reads from `./chroma_db` on every query — it must be run from the repo root so the relative path resolves correctly.

## Data Sources

Scraped via `WebBaseLoader` (plain HTTP — no browser automation):
- **NPS** — `nps.gov/{park-code}/index.htm` and `planyourvisit/weather.htm` pages for Mt Rainier (`mora`), Olympic (`olym`), and North Cascades (`noca`)
- **Wikipedia** — trail, wilderness area, national park, and national forest articles
- **Recreation.gov** — permit pages (e.g. Enchantments, permit ID `233273`)

**Blocked sources (do not retry without a headless browser):**
- WTA (wta.org) — Cloudflare JS challenge; `cloudscraper` is insufficient
- USFS activity pages — JavaScript-rendered, return no content

NPS sub-pages (`hiking.htm`, `day-hikes.htm`, `seasons.htm`) return 404 — use `index.htm` and `weather.htm` instead.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Populate the vector store
Run all cells in `data_ingestion.ipynb` top to bottom. Requires `OPENAI_API_KEY` in a `.env` file. Produces `./chroma_db/`.

### Run the agent
```bash
cd hiking_agent
adk run .
# or for the web UI:
adk web
```

### Add a `.env` file
```
OPENAI_API_KEY=sk-...
```

## Key Configuration

| Setting | Value |
|---|---|
| Embedding model | `text-embedding-3-small` |
| Chroma collection | `washington_hikes` |
| Chroma path | `./chroma_db` (relative to repo root) |
| Agent model | `gemini-2.5-flash` |
| Retriever k | 5 |
| Chunk size | 1000 chars, 150 overlap |

The embedding model and collection name must match exactly between `data_ingestion.ipynb` and `hiking_agent/agent.py` — mismatches produce empty results silently.
