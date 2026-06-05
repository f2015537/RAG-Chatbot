# Washington Hikes RAG Chatbot

A production-style **Retrieval-Augmented Generation (RAG)** chatbot that answers natural language questions about hiking trails in Washington state — difficulty ratings, distances, elevation gain, best season to visit, permit requirements, and current trail conditions.

Built with **LangChain**, **Chroma**, **OpenAI Embeddings**, and **Google's Agent Development Kit (ADK)**.

---

## Demo

```
User:  What's the best time of year to hike the Enchantments?
Agent: The Enchantments are typically accessible from late June through October,
       with peak conditions in August and September when the snow has melted and
       the larches turn golden. An overnight permit via the Recreation.gov lottery
       is required — the advance lottery opens in February, and a daily lottery
       runs May through October...
```

```
User:  Recommend some easy hikes near Seattle suitable for kids.
Agent: A few great options close to Seattle:
       • Rattlesnake Ledge — 4 miles round trip, 1,100 ft gain, rewarding views
         at the top with minimal technical difficulty.
       • Little Si — 5 miles round trip, a gentler version of Mount Si with
         forest trail and a rocky viewpoint...
```

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
git clone https://github.com/your-username/RAG-Chatbot.git
cd RAG-Chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...        # required by Google ADK for Gemini
```

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
