# DocMind RAG - Quick Reference Cheat Sheet

## 🎯 One-Line Summary
**DocMind RAG** = Upload documents → AI reads them → Ask questions → Get answers with citations

---

## 🏗️ Architecture at a Glance

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   FastAPI   │────▶│  LangGraph  │────▶│   GPT-4     │
│   (React)   │     │   (API)     │     │   (RAG)     │     │   (LLM)     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │   Celery    │     │   Qdrant    │
                    │  (Workers)  │     │  (Vectors)  │
                    └─────────────┘     └─────────────┘
```

---

## 📤 Document Upload Pipeline

| Step | What Happens | Technology |
|------|--------------|------------|
| 1. Upload | File saved, task queued | FastAPI, Celery |
| 2. Classify | Detect page types | Page Classifier |
| 3. Extract | Get text/tables/images | PyMuPDF, Unstructured, GPT-4 Vision |
| 4. Chunk | Split into pieces | RecursiveCharacterTextSplitter |
| 5. Embed | Convert to vectors | text-embedding-3-large |
| 6. Store | Save to database | Qdrant |

---

## 🔍 Query Pipeline

| Step | What Happens | Technology |
|------|--------------|------------|
| 1. Cache | Check for similar past queries | Redis Semantic Cache |
| 2. Enhance | Expand acronyms, rewrite | Query Expander |
| 3. Retrieve | Find relevant chunks | Hybrid (Dense + BM25) |
| 4. Validate | Check if docs are relevant | GPT-4o-mini |
| 5. Rerank | Score and sort results | Cross-Encoder |
| 6. Generate | Create answer | GPT-4 |
| 7. Validate | Check if answer is grounded | GPT-4o-mini |

---

## 🛠️ Technology Choices - Quick Reference

### Why This Technology?

| Component | Choice | Why Not Alternatives |
|-----------|--------|---------------------|
| **Vector DB** | Qdrant | Pinecone (expensive), Chroma (not production-ready) |
| **Embeddings** | text-embedding-3-large | 3072 dims = better accuracy than 1536 |
| **LLM** | GPT-4 | Best reasoning, 128K context |
| **Framework** | LangChain + LangGraph | LlamaIndex (less flexible), raw API (too much code) |
| **Graph DB** | Neo4j | Most mature, Cypher language |
| **Cache** | Redis | Fast, supports semantic similarity |
| **Task Queue** | Celery | Python native, reliable |

---

## 📊 Key Numbers

| Parameter | Value | Why |
|-----------|-------|-----|
| Chunk size | 1000 chars | ~250 tokens, good context |
| Chunk overlap | 200 chars | 20%, prevents info loss |
| Embedding dims | 3072 | More nuance |
| Dense:Sparse | 60:40 | Balance meaning + keywords |
| Cache threshold | 92% | Accuracy vs hit rate |
| Max retries | 2 | Prevent infinite loops |
| Top-K retrieval | 5 | Enough context, not too much |

---

## 🎯 Content Type Handling

| Content | Extraction | Chunking | Why |
|---------|------------|----------|-----|
| Text | PyMuPDF | 1000 chars | Fast, accurate |
| Tables | Unstructured | Keep intact | Preserve structure |
| Charts | GPT-4 Vision | Keep intact | Need description |
| Code | PyMuPDF | 1500 chars | Keep functions together |
| Handwriting | Vision + OCR | 500 chars | Noisy, smaller chunks |

---

## 🔄 Self-Correction Flow

```
Query → Enhance → Retrieve → Validate
                      ↓
              Documents relevant?
                 /        \
               YES         NO
                ↓           ↓
            Generate    Retry (max 2)
                ↓
        Answer grounded?
           /        \
         YES         NO
          ↓           ↓
       Return     Retry (max 2)
```

---

## 💡 Key Analogies

| Concept | Analogy |
|---------|---------|
| **RAG** | Brilliant librarian who reads all books |
| **Embeddings** | Magical map where similar things are close |
| **Vector DB** | Smart filing cabinet that organizes by topic |
| **Chunking** | Cutting pizza into slices |
| **Async Processing** | Restaurant kitchen with order board |
| **Page Classification** | Harry Potter's Sorting Hat |
| **Specialist Agents** | Hospital with different departments |
| **Knowledge Graph** | Mind map of your documents |
| **Semantic Cache** | Student checking notes before solving again |

---

## 🚀 Quick Commands

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f backend

# Access
# Frontend: http://localhost:3000
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs

# Upload a document
curl -X POST http://localhost:8000/api/upload -F "file=@document.pdf"

# Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is attention?", "top_k": 5}'
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `backend/app/main.py` | FastAPI entry point |
| `backend/app/chains/agentic_rag.py` | LangGraph RAG pipeline |
| `backend/app/agents/orchestrator.py` | Document processing coordinator |
| `backend/app/ingestion/content_chunker.py` | Chunking strategies |
| `backend/app/vectorstore/store.py` | Qdrant integration |
| `backend/app/retrievers/hybrid_retriever.py` | Dense + BM25 search |
| `frontend/src/App.tsx` | React main component |

---

## ❓ FAQ

**Q: Why not just use ChatGPT?**
A: ChatGPT doesn't know YOUR documents. RAG gives it your data as context.

**Q: Why chunk documents?**
A: AI has token limits. Smaller pieces = more precise retrieval.

**Q: Why hybrid retrieval?**
A: Dense catches synonyms, sparse catches exact terms. Best of both.

**Q: Why validate answers?**
A: Prevents hallucinations. Ensures answers are grounded in sources.

**Q: Why Neo4j?**
A: For relationship questions that vector search can't answer.

---

## 📚 Learn More

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Full technical details
- [VIDEO_SERIES_OUTLINE.md](./VIDEO_SERIES_OUTLINE.md) - Tutorial videos
- [SPEAKER_NOTES.md](./SPEAKER_NOTES.md) - Detailed explanations
- [presentation.html](./presentation.html) - Interactive slides
