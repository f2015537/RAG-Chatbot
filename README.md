# Washington Hikes RAG Chatbot

A production-style **Retrieval-Augmented Generation (RAG)** chatbot that answers natural language questions about hiking trails in Washington state — difficulty ratings, distances, elevation gain, best season to visit, permit requirements, and current trail conditions.

Built with **LangChain**, **Chroma**, **OpenAI Embeddings**, and **Google's Agent Development Kit (ADK)**.

---

## Demo

The following exchanges are from a live session on `adk web`.

---

**What's the hardest hike in Washington state?**

> The Wonderland Trail, which circumnavigates Mount Rainier, is one of the more challenging hikes in Washington state. It's approximately 93 miles long with a cumulative 22,000 feet of elevation gain, taking an average of 10 to 14 days to complete. The trail goes over many ridges of Mount Rainier and passes through various life zones from lowland forests to subalpine meadows.

---

**What are the current trail conditions in North Cascades?**

> The best weather for visiting the North Cascades is generally between mid-June and late-September, with snow typically off all but the highest trails by July. Spring and autumn are also popular for scenic drives and wildlife viewing in the valleys, though storms with rain and wind are common, requiring warm, waterproof clothing and a tent for backcountry trips.
>
> Heavy snow and rain, especially at high elevations, characterize winters, with avalanches common in winter and spring. The east side of the Cascades (e.g., Stehekin) tends to be drier and warmer in summer, with temperatures reaching into the 90s, while winter brings snow cover to elevations above 2,000 feet from late fall into spring.
>
> For the most up-to-date, trail-by-trail conditions, it's recommended to check the official North Cascades National Park website.

---

**Hikes suitable for kids on the Olympic Peninsula?**

> For hikes suitable for kids on the Olympic Peninsula, consider these options:
>
> **Ozette Loop** — This 9-mile loop offers a 3-mile section with a boardwalk through a coastal cedar swamp, which is generally easy walking. The trail also includes a 3-mile walk along the ocean. The boardwalk-enhanced sections make it more accessible.
>
> **Hurricane Hill Trail** — Located near the Hurricane Ridge visitor center, this is a paved trail about 1.6 miles long each way, with an elevation gain of about 700 feet. Its paved surface and relatively short length make it a good option for families with children. Be aware that snow can be present on the trails even as late as July.

---

> **Honest limitations:** Queries requiring specific per-trail data (exact distances, difficulty ratings for individual hikes) sometimes return partial answers because the current knowledge base is built from regional/park-level sources rather than per-trail databases. See [Potential Improvements](#potential-improvements) for how this would be addressed.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│                   (data_ingestion.ipynb)                        │
│                                                                 │
│  Web Sources          LangChain              ChromaDB           │
│  ───────────      ──────────────────      ──────────────        │
│  NPS.gov      →   WebBaseLoader       →   text-embedding   →   │
│  Wikipedia        BeautifulSoup           -3-small              │
│  Recreation.gov   RecursiveCharacter      ./chroma_db/          │
│                   TextSplitter            washington_hikes       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ persisted vector store
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AGENT RUNTIME                             │
│                    (hiking_agent/)                              │
│                                                                 │
│   User Query                                                    │
│       │                                                         │
│       ▼                                                         │
│   Google ADK Agent  ──→  retrieve_info()  ──→  Chroma          │
│   (gemini-2.5-flash)      FunctionTool         similarity       │
│       │                   k=5 results          search           │
│       ▼                                                         │
│   Grounded Response                                             │
└─────────────────────────────────────────────────────────────────┘
```

The pipeline is intentionally split into two stages:

- **Ingestion** runs once (or whenever the data needs refreshing) and is completely independent of the agent
- **Agent** is stateless — it reconnects to the persisted vector store on every query, making it trivially deployable

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | [Google ADK](https://google.github.io/adk-docs/) |
| LLM | Gemini 2.5 Flash |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | [Chroma](https://www.trychroma.com/) (local persistent) |
| Document loading | LangChain `WebBaseLoader` + BeautifulSoup |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Ingestion runtime | Jupyter Notebook |

---

## Data Sources

The ingestion pipeline scrapes **29 URLs** across three authoritative sources:

### National Park Service
Main and plan-your-visit pages for the three major Washington national parks, plus dedicated **weather/seasons pages** that contain explicit guidance on the best time to visit:
- Mount Rainier National Park (`mora`)
- Olympic National Park (`olym`)
- North Cascades National Park (`noca`)

### Wikipedia
Rich articles (2K–83K chars each) covering individual trails, wilderness areas, national parks, and national forests:

| Category | Articles |
|---|---|
| Iconic trails | Wonderland Trail, Pacific Crest Trail, The Enchantments |
| Individual trails | Mount Si, Rattlesnake Ledge, North Cascades Highway (Maple Pass) |
| Wilderness areas | Alpine Lakes, Glacier Peak, Goat Rocks, Pasayten |
| National parks | Mt Rainier, Olympic, North Cascades |
| National forests | Mt Baker-Snoqualmie, Okanogan-Wenatchee, Olympic |
| Regions | Olympic Peninsula, North Cascades, Issaquah Alps |

### Recreation.gov
Permit pages with seasonal access windows and quota information (e.g. the Enchantments lottery system).

> **Note on source selection:** Washington Trails Association (WTA) — the most comprehensive per-trail database — is protected by Cloudflare's JS challenge and cannot be reliably scraped without a full headless browser. USFS activity pages are JavaScript-rendered. Both were excluded in favour of sources that load reliably with a standard HTTP client.

---

## Project Structure

```
RAG-Chatbot/
├── data_ingestion.ipynb     # Scraping, chunking, embedding, and Chroma ingestion
├── hiking_agent/
│   ├── __init__.py
│   └── agent.py             # Google ADK agent with RAG retrieval tool
├── requirements.txt
├── .env                     # Not committed — see setup below
├── .gitignore
└── chroma_db/               # Not committed — generated by the notebook
```

---

## Local Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/f2015537/RAG-Chatbot.git
cd RAG-Chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Two `.env` files are required — one for the ingestion notebook and one for the ADK agent:

```bash
# Root — used by data_ingestion.ipynb
cp .env.example .env

# Agent — used by the Google ADK runtime
cp hiking_agent/.env.example hiking_agent/.env
```

| File | Keys |
|---|---|
| `.env` | `OPENAI_API_KEY` |
| `hiking_agent/.env` | `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_GENAI_USE_VERTEXAI` |

> Set `GOOGLE_GENAI_USE_VERTEXAI=0` to use the Gemini API directly (no Vertex AI setup required).

### 3. Populate the vector store

Open `data_ingestion.ipynb` in Jupyter and run all cells top to bottom:

```bash
jupyter notebook data_ingestion.ipynb
```

This will:
1. Scrape all 29 URLs
2. Filter and chunk the content into 640 documents
3. Embed each chunk with `text-embedding-3-small`
4. Persist the Chroma collection to `./chroma_db/`

Expected output: `Stored 640 chunks in ./chroma_db`

### 4. Run the agent

```bash
# Interactive CLI
adk run hiking_agent

# Browser-based UI
adk web
```

---

## Design Decisions

### Why RAG instead of fine-tuning?
Trail conditions, permit availability, and access windows change seasonally. RAG allows the knowledge base to be updated by re-running the ingestion pipeline without touching the model — fine-tuning would require a full retraining cycle for every data refresh.

### Why Chroma (local) instead of a hosted vector DB?
For a single-developer project and showcase purposes, a local persistent Chroma store eliminates infrastructure cost and setup friction. The retrieval interface (`as_retriever()`) is identical to hosted options like Pinecone or Weaviate — swapping the backend requires changing a single constructor call.

### Why `text-embedding-3-small`?
It offers the best cost-to-quality ratio for English retrieval tasks at this scale. `text-embedding-3-large` improves recall marginally but costs 5× more per token — not justified for a 640-chunk corpus.

### Why split ingestion and agent into separate components?
The ingestion pipeline is a one-shot operation that can be scheduled or triggered independently of the running agent. This separation means the agent can be deployed without any scraping dependencies, and the data can be refreshed without redeploying the agent.

### Chunk size: 1000 chars / 150 overlap
Tested against the source material: NPS and Wikipedia content benefits from slightly larger chunks (more context per retrieval hit) without exceeding the token budget for the 5 retrieved chunks passed to Gemini. The 150-char overlap prevents information loss at chunk boundaries.

---

## Potential Improvements

- **Playwright-based scraper** to unlock WTA's per-trail pages (difficulty, distance, elevation for 10,000+ hikes) — the single highest-impact data improvement
- **Scheduled re-ingestion** to keep trail conditions and permit availability current
- **Metadata filtering** — store region, difficulty, and distance as Chroma metadata fields to enable structured pre-filtering before semantic search
- **Hybrid search** — combine dense vector search with BM25 sparse retrieval for better recall on specific trail names
- **Evaluation harness** — a small golden dataset of Q&A pairs to measure retrieval precision and answer quality across data refreshes
