# DocMind RAG - Video Tutorial Series Outline

## 🎬 Complete Video Series Structure

This tutorial series is designed for team members with **no prior knowledge** of RAG, LLMs, or vector databases. Each video builds on the previous one.

---

## Video 1: Introduction - What Problem Are We Solving? (10-15 mins)

### Script Outline

**Opening Hook (1 min)**
> "Imagine you have 10,000 PDF documents - research papers, contracts, manuals. Someone asks: 'What did we agree about pricing in the 2023 contracts?' How do you find that answer? Ctrl+F won't work across thousands of files. This is the problem DocMind RAG solves."

**The Problem Explained (3 mins)**
- Traditional search (Ctrl+F, grep) only finds exact matches
- Can't understand meaning or context
- Can't synthesize information from multiple documents
- Can't answer questions in natural language

**Real-World Analogy: The Library Analogy (2 mins)**
> "Think of a massive library with millions of books. Traditional search is like having an index that only lists exact words. RAG is like having a brilliant librarian who:
> 1. Understands what you're really asking
> 2. Knows which books are relevant (even if they don't use your exact words)
> 3. Reads the relevant sections
> 4. Gives you a synthesized answer with page references"

**What is RAG? (3 mins)**
- **R**etrieval - Find relevant information
- **A**ugmented - Enhance the AI's knowledge
- **G**eneration - Create a natural language answer

**Demo: Before vs After (3 mins)**
- Show searching manually through PDFs
- Show asking DocMind the same question
- Highlight: speed, accuracy, citations

**Architecture Overview (2 mins)**
- High-level diagram: Frontend → Backend → AI → Database
- Preview of what we'll cover in the series

---

## Video 2: The Document Upload Journey (15-20 mins)

### Script Outline

**Opening (1 min)**
> "When you upload a PDF, a lot happens behind the scenes. Let's follow a document's journey from upload to being searchable."

**Step 1: File Upload (2 mins)**
- Frontend sends file to backend
- File validation (type, size)
- Saved to temporary storage

**Step 2: Why We Need Async Processing (3 mins)**
> "Imagine a restaurant. If the chef had to cook each order completely before taking the next one, customers would wait forever. Instead, orders go to a queue, and multiple chefs work in parallel."

- **Celery** = The kitchen staff (workers)
- **Redis** = The order board (message queue)
- Why: Large PDFs can take minutes to process

**Step 3: Page Classification - The Sorting Hat (4 mins)**
> "Like Harry Potter's Sorting Hat, we examine each page and decide: 'Is this text? A table? A chart? Code?'"

Why different types matter:
- **Text**: Easy to extract, standard chunking
- **Tables**: Must keep rows/columns together (splitting = data loss)
- **Charts/Images**: Need AI vision to "see" and describe
- **Code**: Need to keep functions together

**Step 4: Specialist Agents - The Expert Team (4 mins)**
> "Think of a hospital. You don't send everyone to the same doctor. A broken bone goes to orthopedics, heart issues to cardiology. Similarly, we have specialist agents."

- **TableSpecialist**: Extracts tables, merges multi-page tables
- **ChartSpecialist**: Uses GPT-4 Vision to describe charts
- **CodeSpecialist**: Preserves code structure
- **TextSpecialist**: Handles regular text

**Step 5: Chunking - Breaking Into Searchable Pieces (3 mins)**
> "You can't swallow a whole pizza. You cut it into slices. Similarly, we cut documents into 'chunks' that AI can process."

Why chunk?
- AI has token limits (like a stomach has capacity)
- Smaller pieces = more precise retrieval
- Overlap prevents losing context at boundaries

**Demo: Upload a PDF and show the processing stages**

---

## Video 3: Embeddings - Teaching AI to Understand Meaning (15-20 mins)

### Script Outline

**Opening Question (1 min)**
> "How does a computer understand that 'car' and 'automobile' mean the same thing? Or that 'bank' (river) and 'bank' (money) are different? The answer: Embeddings."

**The Problem with Keywords (2 mins)**
- "Happy" vs "Joyful" - same meaning, different words
- "Apple" - fruit or company?
- Computers see text as random characters

**What Are Embeddings? (4 mins)**
> "Imagine a magical map where similar things are placed close together. On this map:
> - 'King' and 'Queen' are close (both royalty)
> - 'King' and 'Banana' are far apart
> - 'Paris' is close to 'France' and 'Eiffel Tower'"

- Embeddings = coordinates on this map
- Each word/sentence gets a position (vector)
- Similar meanings = close positions

**Visual Demo: 2D Embedding Space (3 mins)**
- Show words plotted on a 2D graph
- Demonstrate: "dog" near "puppy", far from "computer"
- Explain: Real embeddings have 3072 dimensions (not just 2)

**Why text-embedding-3-large? (3 mins)**

| Model | Dimensions | Quality | Speed | Cost |
|-------|------------|---------|-------|------|
| text-embedding-3-small | 1536 | Good | Fast | Cheap |
| text-embedding-3-large | 3072 | Best | Medium | Medium |
| text-embedding-ada-002 | 1536 | Good | Fast | Cheap |

> "More dimensions = more nuance. Like describing a color with RGB (3 values) vs describing with RGB + brightness + saturation (5 values). More detail = better matching."

**The Embedding Process (3 mins)**
1. Take a chunk of text
2. Send to OpenAI API
3. Get back 3072 numbers
4. Store these numbers in vector database

**Why Not Use Free/Local Models? (2 mins)**
- Quality matters for retrieval accuracy
- OpenAI embeddings are state-of-the-art
- Alternatives: Sentence-Transformers (free, local, but less accurate)

**Demo: Show embedding similarity scores**

---

## Video 4: Vector Database (Qdrant) - The Smart Filing Cabinet (15 mins)

### Script Outline

**Opening Analogy (2 mins)**
> "Imagine a filing cabinet where files automatically arrange themselves by topic. Ask for 'vacation policies' and it instantly finds all related documents - even ones titled 'PTO guidelines' or 'time-off procedures'. That's a vector database."

**Why Not Regular Databases? (3 mins)**

| Feature | SQL Database | Vector Database |
|---------|--------------|-----------------|
| Search type | Exact match | Similarity match |
| Query | "WHERE title = 'vacation'" | "Find similar to 'vacation'" |
| Understands meaning | ❌ No | ✅ Yes |
| Handles typos | ❌ No | ✅ Yes |
| Finds synonyms | ❌ No | ✅ Yes |

**How Vector Search Works (4 mins)**
1. Your query gets embedded (converted to numbers)
2. Database finds vectors closest to your query
3. Returns the original text chunks

> "It's like GPS finding the nearest restaurants. Your location = query embedding. Restaurants = document embeddings. Distance = similarity."

**Why Qdrant? (3 mins)**

| Vector DB | Pros | Cons |
|-----------|------|------|
| **Qdrant** | Fast, scalable, filtering, open-source | Newer |
| Pinecone | Managed, easy | Expensive, vendor lock-in |
| Weaviate | GraphQL, hybrid | Complex setup |
| Chroma | Simple, local | Not production-ready |
| FAISS | Fast, Facebook | No persistence, complex |

> "We chose Qdrant because: open-source (no vendor lock-in), fast, supports filtering by metadata, and scales well."

**Demo: Vector search in action**
- Show query → embedding → search → results
- Show similarity scores

---

## Video 5: The RAG Pipeline - Putting It All Together (20 mins)

### Script Outline

**Opening (1 min)**
> "Now we understand the pieces. Let's see how they work together when you ask a question."

**The Query Journey - Step by Step (15 mins)**

**Step 1: Cache Check (2 mins)**
> "Before doing any work, we check: 'Have we answered this before?' Like a student checking their notes before solving a problem again."

- Semantic cache (Redis) stores previous Q&A
- If similar question (>92% match) → return cached answer
- Saves time and money (no API calls)

**Step 2: Query Enhancement (2 mins)**
> "Sometimes questions need clarification. 'What's CoT?' becomes 'What is Chain of Thought prompting in large language models?'"

- Expand acronyms
- Rewrite for clarity
- Add context

**Step 3: Hybrid Retrieval (3 mins)**
> "We use two search methods and combine them - like asking both Google AND a librarian."

- **Dense search (60%)**: Meaning-based (embeddings)
- **Sparse search (40%)**: Keyword-based (BM25)
- **Why both?** Dense catches synonyms, sparse catches exact terms

**Step 4: Reranking (2 mins)**
> "Initial search gives us candidates. Reranking is like a second opinion - a more careful look at each result."

- Cross-encoder model scores each (query, document) pair
- More accurate but slower
- Only applied to top candidates

**Step 5: Validation Loop (2 mins)**
> "Before answering, we ask: 'Are these documents actually relevant?' If not, we try again with a better query."

- LLM checks relevance
- Up to 2 retries
- Prevents hallucination

**Step 6: Answer Generation (2 mins)**
- Combine: query + documents + conversation history
- Send to GPT-4
- Generate answer with citations

**Step 7: Answer Validation (2 mins)**
> "Final check: 'Is this answer grounded in the sources?' We don't want the AI making things up."

**Demo: Trace a query through the entire pipeline**

---

## Video 6: Why LangChain & LangGraph? (15 mins)

### Script Outline

**Opening Question (1 min)**
> "Why use a framework? Can't we just call OpenAI directly?"

**The Problem Without Frameworks (3 mins)**
```python
# Without LangChain - lots of boilerplate
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
# Now handle: retries, streaming, different models, 
# prompt templates, output parsing, chains...
```

**What LangChain Provides (4 mins)**
- **Document Loaders**: PDF, DOCX, HTML, CSV... (50+ formats)
- **Text Splitters**: Smart chunking strategies
- **Embeddings**: Unified interface for any provider
- **Vector Stores**: Same code works with Qdrant, Pinecone, etc.
- **Chains**: Connect components together
- **Prompts**: Reusable templates

> "It's like using React instead of vanilla JavaScript. You CAN do everything manually, but why?"

**Why LangGraph for RAG? (4 mins)**
> "LangChain is great for simple chains. But RAG needs loops, conditions, retries. That's where LangGraph shines."

```
LangChain: A → B → C (linear)
LangGraph: A → B → (if bad) → A again → B → C (stateful, loops)
```

- **State management**: Track query, documents, retries
- **Conditional edges**: Different paths based on validation
- **Self-correction**: Retry if results are bad

**Alternatives Comparison (3 mins)**

| Framework | Pros | Cons |
|-----------|------|------|
| **LangChain** | Comprehensive, popular | Can be complex |
| LlamaIndex | Great for RAG specifically | Less flexible |
| Haystack | Good for search | Smaller community |
| Raw API calls | Full control | Lots of code |

---

## Video 7: Knowledge Graph (Neo4j) - Understanding Relationships (15 mins)

### Script Outline

**Opening Question (1 min)**
> "Vector search finds similar documents. But what if you need to understand relationships? 'Who reports to the CEO?' requires understanding org structure, not just similarity."

**The Limitation of Vector Search (3 mins)**
- Great for: "Find documents about X"
- Bad for: "What's connected to X?"
- Example: "What technologies does Transformer use?"
  - Vector search: finds documents mentioning Transformer
  - Knowledge graph: Transformer → uses → Attention → uses → Softmax

**What is a Knowledge Graph? (3 mins)**
> "A knowledge graph is like a mind map of your documents. Entities (things) connected by relationships."

```
[Transformer] --uses--> [Attention]
[Attention] --has_type--> [Self-Attention]
[BERT] --based_on--> [Transformer]
[GPT] --based_on--> [Transformer]
```

**Why Neo4j? (3 mins)**

| Graph DB | Pros | Cons |
|----------|------|------|
| **Neo4j** | Mature, Cypher query language, visualization | Resource heavy |
| Amazon Neptune | Managed, scalable | AWS lock-in |
| ArangoDB | Multi-model | Less graph-focused |
| NetworkX | Simple, Python | Not persistent |

**How We Use It (3 mins)**
1. Extract entities from documents (LLM)
2. Extract relationships between entities
3. Store in Neo4j
4. During query: expand context with related entities

**When is it Useful? (2 mins)**
- Multi-hop questions: "What do BERT and GPT have in common?"
- Relationship queries: "What technologies are used in Transformers?"
- Exploration: "Show me everything related to Attention"

**Demo: Knowledge graph visualization**

---

## Video 8: Caching & Performance Optimization (10 mins)

### Script Outline

**Opening (1 min)**
> "Every API call costs money and time. Smart caching can reduce both by 80%."

**Semantic Cache Explained (3 mins)**
> "Traditional cache: exact match only. Semantic cache: similar questions get cached answers."

- "What is attention?" → cached
- "Explain attention mechanism" → 95% similar → cache hit!
- "How does attention work in transformers?" → 89% similar → cache miss

**Why 92% Threshold? (2 mins)**
- Too low (80%): Wrong answers returned
- Too high (99%): Almost no cache hits
- 92%: Sweet spot for accuracy vs savings

**Redis for Speed (2 mins)**
- In-memory = microsecond access
- Persistence = survives restarts
- TTL = automatic expiration (24 hours)

**Cost Savings Example (2 mins)**
```
Without cache:
- 1000 queries/day × $0.01/query = $10/day = $300/month

With 70% cache hit rate:
- 300 queries/day × $0.01/query = $3/day = $90/month
- Savings: $210/month (70%)
```

---

## Video 9: Multi-Turn Conversations (10 mins)

### Script Outline

**The Problem (2 mins)**
> "User asks: 'What is attention?' Then: 'How is it different from RNNs?' The AI needs to know 'it' refers to attention."

**Conversation Memory (3 mins)**
- Store conversation history in PostgreSQL
- Include last 3 turns in context
- AI understands references

**Example Flow (3 mins)**
```
Turn 1: "What is attention?"
→ AI explains attention mechanism

Turn 2: "How is it different from RNNs?"
→ AI knows "it" = attention
→ Compares attention vs RNNs

Turn 3: "Show me the formula"
→ AI knows we're discussing attention
→ Shows: Attention(Q,K,V) = softmax(QK^T/√d)V
```

**Why PostgreSQL? (2 mins)**
- Relational data (users, conversations, messages)
- ACID compliance (data integrity)
- Familiar, reliable, scalable

---

## Video 10: Production Deployment & Monitoring (15 mins)

### Script Outline

**Architecture for Production (5 mins)**
- Docker containers for each service
- Load balancing for API
- Separate workers for processing
- Managed databases for reliability

**Monitoring with LangSmith (5 mins)**
- Trace every query through the pipeline
- See: latency, tokens used, errors
- Debug failed queries
- A/B test prompts

**Cost Management (3 mins)**
- Monitor API usage
- Set rate limits
- Use caching aggressively
- Choose right model for each task

**Security Considerations (2 mins)**
- API key management
- Data encryption
- Access control
- Audit logging

---

## 📁 Supporting Materials to Create

1. **Slide Deck** (PowerPoint/Google Slides)
2. **Interactive Diagrams** (HTML/Mermaid)
3. **Code Walkthrough Notebooks** (Jupyter)
4. **Cheat Sheet** (1-page PDF)
5. **Glossary of Terms**
6. **FAQ Document**

---

## 🎯 Recording Tips

1. **Use screen recording** with face cam in corner
2. **Highlight cursor** for visibility
3. **Use animations** for diagrams (reveal step by step)
4. **Include timestamps** in video description
5. **Add captions** for accessibility
6. **Keep videos under 20 mins** (attention span)
7. **End each video with preview** of next topic

---

## 📅 Suggested Recording Order

1. Video 1: Introduction (sets context)
2. Video 3: Embeddings (foundational concept)
3. Video 4: Vector Database (builds on embeddings)
4. Video 2: Document Upload (uses embeddings + vector DB)
5. Video 5: RAG Pipeline (combines everything)
6. Video 6: LangChain/LangGraph (explains framework choice)
7. Video 7: Knowledge Graph (advanced feature)
8. Video 8: Caching (optimization)
9. Video 9: Multi-Turn (feature)
10. Video 10: Production (deployment)
