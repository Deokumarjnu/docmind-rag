# DocMind RAG

Enterprise Document Intelligence Platform with RAG (Retrieval Augmented Generation)

## Overview

DocMind RAG is a production-ready document intelligence platform built with:

- **LangChain** for document loaders, text splitters, embeddings, and vector stores
- **LangGraph** for durable execution, state management, and complex RAG workflows
- **Deep Agents** for intelligent document processing orchestration with specialized subagents
- **Agentic RAG with Self-Correction** pattern for handling complex queries

## Architecture Overview (Full LangChain Ecosystem)

File formate supported 
• PDF 
• DOCX 
• DOC 
• HTML 
• TXT 
• CSV 
• XLSX 
• XLS 
• JSON 
• JSONL 
• Markdown

After reviewing the latest LangChain documentation, we leverage their complete ecosystem:

- **LangChain** for document loaders, text splitters, embeddings, and vector stores
- **LangGraph** for durable execution, state management, and complex RAG workflows
- **Deep Agents** for intelligent document processing orchestration with specialized subagents
- **Agentic RAG with Self-Correction** pattern for handling complex queries

```mermaid
flowchart TB
    subgraph ingestion [Document Ingestion Pipeline]
        Upload[File Upload API]
        DocLoader[LangChain Document Loaders]
        TextSplitter[Text Splitters]
        EmbedModel[Embedding Model]
        VectorStore[Vector Store]
    end
    
    subgraph retrieval [LangGraph RAG Agent]
        QueryEnhance[Query Enhancement]
        HybridRetriever[Hybrid Retriever]
        RetrievalValidation[Retrieval Validation]
        ContextBuilder[Context Assembly]
    end
    
    subgraph generation [Response Generation]
        LLM[LLM with Structured Output]
        AnswerValidation[Answer Validation]
        SelfCorrection[Self-Correction Loop]
        Citations[Citation Extraction]
    end
    
    Upload --> DocLoader
    DocLoader --> TextSplitter
    TextSplitter --> EmbedModel
    EmbedModel --> VectorStore
    
    Query[User Query] --> QueryEnhance
    QueryEnhance --> HybridRetriever
    HybridRetriever --> VectorStore
    HybridRetriever --> RetrievalValidation
    RetrievalValidation -->|"Not Relevant"| QueryEnhance
    RetrievalValidation -->|"Relevant"| ContextBuilder
    ContextBuilder --> LLM
    LLM --> AnswerValidation
    AnswerValidation -->|"Invalid"| SelfCorrection
    SelfCorrection --> HybridRetriever
    AnswerValidation -->|"Valid"| Citations
    Citations --> Response[Answer with Sources]
```

### Deep Agents Architecture (Document Processing Orchestration)

Deep Agents provide intelligent orchestration for complex, multi-step document processing tasks. Instead of processing pages linearly, a main orchestrator agent delegates to specialized subagents based on content type.

```mermaid
flowchart TB
    subgraph upload [Document Upload]
        PDF[PDF Document]
        PageSplit[Page Splitter]
    end
    
    subgraph deepagent [Deep Agent Orchestrator]
        MainAgent[Main Document Agent]
        Classifier[Content Classifier]
        
        subgraph subagents [Specialized Subagents]
            TableAgent[Table Specialist]
            CodeAgent[Code Specialist]
            HandwritingAgent[Handwriting Specialist]
            ChartAgent[Chart/Diagram Specialist]
            TextAgent[Text Specialist]
        end
        
        MainAgent --> Classifier
        Classifier -->|"Tables"| TableAgent
        Classifier -->|"Code"| CodeAgent
        Classifier -->|"Handwriting"| HandwritingAgent
        Classifier -->|"Charts/Diagrams"| ChartAgent
        Classifier -->|"Plain Text"| TextAgent
    end
    
    subgraph storage [Storage Layer]
        Chunker[Content-Aware Chunker]
        Embedder[Embedding Generator]
        VectorStore[Vector Store]
    end
    
    PDF --> PageSplit
    PageSplit --> MainAgent
    TableAgent --> Chunker
    CodeAgent --> Chunker
    HandwritingAgent --> Chunker
    ChartAgent --> Chunker
    TextAgent --> Chunker
    Chunker --> Embedder --> VectorStore
```

#### Why Deep Agents for Document Processing?

| Benefit | Description |
|---------|-------------|
| **Task Decomposition** | Complex 500+ page PDFs are broken into manageable subtasks |
| **Specialized Expertise** | Each subagent is optimized for its content type |
| **Context Isolation** | Subagents maintain focused context, avoiding confusion |
| **Parallel Processing** | Multiple subagents can work simultaneously |
| **Long-term Memory** | Processing state persists across sessions |
| **Adaptive Routing** | Main agent learns which subagent handles what best |

### Page-Level Adaptive PDF Ingestion (Critical for Large Documents)

This is the core innovation for handling large PDFs efficiently. Never process 500 pages at once!

```mermaid
flowchart TB
    PDF[PDF 500+ pages] --> PageSplit[Page Splitter]
    PageSplit --> Classifier[Page Type Classifier]
    
    Classifier --> TextPage[Text Page]
    Classifier --> TablePage[Table Page]
    Classifier --> ImagePage[Image Page]
    Classifier --> MixedPage[Mixed Page]
    
    TextPage --> PyMuPDF[PyMuPDF Fast Extract]
    TablePage --> UnstructuredElements[Unstructured Elements Mode]
    ImagePage --> OCR[OCR Only Strategy]
    MixedPage --> UnstructuredAuto[Unstructured Auto]
    
    PyMuPDF --> Normalize[Normalize + Metadata]
    UnstructuredElements --> Normalize
    OCR --> Normalize
    UnstructuredAuto --> Normalize
    
    Normalize --> ContentChunker[Content-Aware Chunking]
    ContentChunker --> Embed[Embed + Store]
```

#### Extraction Strategies by Page Type

| Page Type | Extractor | Rationale |
|-----------|-----------|-----------|
| Text only | PyMuPDF | Fast, accurate for pure text |
| Tables | Unstructured (elements) | Preserves table structure |
| Image-only | OCR (Unstructured/Cloud) | Only option for scanned pages |
| Mixed | Unstructured (auto) | Handles complex layouts |

#### Content-Aware Chunking

| Content Type | Chunk Strategy |
|--------------|----------------|
| Text | 1000 chars (~250 tokens) with 200 char overlap |
| Tables | Whole table (no splitting) |
| OCR text | 500 chars (smaller for noisy content) |
| Code | AST-based (by function/class) |
| Visual | Single chunk with description |

## Features

### Document Processing
- **Adaptive Page Extraction**: Automatically classifies pages as text, tables, images, charts, code, or handwriting
- **Deep Agent Orchestration**: Specialized subagents for different content types
- **Multi-page Table Merging**: Detects and merges tables spanning multiple pages
- **Vision AI Integration**: GPT-4o for charts, diagrams, and handwritten text
- **Code-aware Chunking**: AST-based chunking for programming code
- **Header/Footer Removal**: Automatic detection and removal of repetitive elements
- **Cross-page Text Continuity**: Handles text that continues across page boundaries

### RAG Pipeline
- **Hybrid Retrieval**: Combines dense embeddings with BM25 sparse search
- **Cross-encoder Reranking**: Improves precision with semantic reranking
- **Query Enhancement**: Acronym expansion and query rewriting
- **Self-correction Loop**: Validates retrieval and answer quality

### Production Features
- **Async Processing**: Celery tasks with progress tracking for large documents
- **Streaming Responses**: Server-Sent Events for real-time answer generation
- **Document Management**: CRUD operations for document lifecycle
- **LangSmith Integration**: Built-in observability and debugging

## Architecture

```
docmind-rag/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application
│   │   ├── config.py               # Configuration settings
│   │   ├── api/                    # API endpoints (routes, schemas, upload)
│   │   ├── agents/                 # Deep Agent orchestrator and specialists
│   │   │   ├── orchestrator.py     # Main document agent coordinator
│   │   │   ├── table_specialist.py # Table extraction & merging
│   │   │   ├── chart_specialist.py # Chart/diagram analysis (GPT-4o Vision)
│   │   │   ├── code_specialist.py  # Code extraction & AST chunking
│   │   │   ├── handwriting_specialist.py  # OCR + Vision
│   │   │   ├── text_specialist.py  # Structure detection
│   │   │   └── tools/              # Agent tools (code, table, text, vision)
│   │   ├── chains/                 # RAG chains
│   │   │   ├── agentic_rag.py      # LangGraph self-correcting RAG
│   │   │   └── rag_chain.py        # Simple LangChain RAG
│   │   ├── ingestion/              # Document processing pipeline
│   │   │   ├── adaptive_extractor.py     # Page-level adaptive extraction
│   │   │   ├── content_chunker.py        # Content-aware chunking
│   │   │   ├── page_classifier.py        # Text/table/image classification
│   │   │   ├── vision_processor.py       # GPT-4o Vision processing
│   │   │   ├── table_merger.py           # Multi-page table merging
│   │   │   ├── code_chunker.py           # AST-based code chunking
│   │   │   └── header_footer_remover.py  # Header/footer detection
│   │   ├── retrievers/             # Hybrid retrieval and reranking
│   │   │   ├── hybrid_retriever.py # Dense + BM25 fusion
│   │   │   ├── query_expander.py   # Query enhancement
│   │   │   └── reranker.py         # Cross-encoder reranking
│   │   ├── vectorstore/            # Qdrant vector store integration
│   │   │   ├── store.py            # Vector store operations
│   │   │   └── document_manager.py # Document CRUD operations
│   │   └── workers/                # Celery async tasks
│   │       ├── celery_app.py       # Celery configuration
│   │       └── tasks.py            # Async processing tasks
│   ├── requirements.txt
│   ├── env.example                 # Environment template
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Main application
│   │   └── components/
│   │       ├── ChatInterface.tsx   # Chat UI with streaming
│   │       ├── FileUpload.tsx      # Document upload with progress
│   │       ├── DocumentList.tsx    # Document management
│   │       ├── SourceViewer.tsx    # Source citation display
│   │       └── UploadProgress.tsx  # Upload progress indicator
│   ├── package.json
│   └── Dockerfile
├── test_docs/                      # Test documents (arXiv papers)
└── docker-compose.yml
```

## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key

### Running with Docker Compose

1. Clone the repository:
```bash
git clone https://github.com/yourusername/docmind-rag.git
cd docmind-rag
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key-here
```

3. Start all services:
```bash
docker-compose up -d
```

4. Access the application:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Local Development

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your-api-key

# Start Qdrant (requires Docker)
docker run -p 6333:6333 qdrant/qdrant

# Start Redis (requires Docker)
docker run -p 6379:6379 redis:alpine

# Run the API server
uvicorn app.main:app --reload

# In another terminal, start Celery worker
celery -A app.workers.celery_app worker --loglevel=info
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## API Endpoints

### Document Upload
```bash
# Async upload with progress tracking
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"

# Check progress
curl http://localhost:8000/api/upload/status/{task_id}
```

### Query
```bash
# Agentic RAG query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 5}'

# Streaming query
curl -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the architecture"}'
```

### Documents
```bash
# List all documents
curl http://localhost:8000/api/documents

# Delete a document
curl -X DELETE http://localhost:8000/api/documents/{document_id}
```

## Configuration

Environment variables can be set in `backend/.env`:

```env
# Required
OPENAI_API_KEY=your-api-key

# Vector Store
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=docmind_documents

# Redis (for Celery async tasks)
REDIS_URL=redis://localhost:6379/0

# LLM Settings (GPT-5.2: 400K context, better reasoning, fewer hallucinations)
LLM_MODEL=gpt-5.2
VISION_MODEL=gpt-5.2
FAST_LLM_MODEL=gpt-4o-mini  # For quick validation/expansion tasks

# Embedding Settings
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072

# Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_PARALLEL_WORKERS=8
MAX_UPLOAD_SIZE_MB=100

# LangSmith (optional)
LANGSMITH_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=docmind-rag
```

## Subagent Responsibilities

Each specialist subagent is optimized for its content type:

```python
# Table Specialist Subagent
table_specialist = create_agent(
    tools=[
        extract_table_structure,
        detect_table_continuation,
        merge_multipage_tables,
        format_table_to_markdown,
    ],
    system_prompt="""You are a Table Extraction Specialist.
    1. Identify table boundaries on the page
    2. Detect if tables continue from previous pages
    3. Merge multi-page tables into complete units
    4. Convert tables to clean markdown format"""
)

# Code Specialist Subagent
code_specialist = create_agent(
    tools=[
        detect_programming_language,
        extract_code_blocks,
        parse_with_ast,
        chunk_by_functions,
    ],
    system_prompt="""You are a Code Extraction Specialist.
    1. Identify code blocks in the document
    2. Detect the programming language
    3. Extract code while preserving formatting
    4. Split code by logical units (functions, classes)"""
)

# Handwriting Specialist Subagent (Vision-capable)
handwriting_specialist = create_agent(
    model="gpt-4o",
    tools=[
        detect_handwritten_regions,
        transcribe_handwriting,
        validate_transcription,
    ],
    system_prompt="""You are a Handwriting Transcription Specialist.
    1. Identify handwritten regions on pages
    2. Accurately transcribe handwritten text
    3. Mark unclear portions with [unclear: ...]"""
)

# Chart/Diagram Specialist Subagent (Vision-capable)
chart_specialist = create_agent(
    model="gpt-4o",
    tools=[
        classify_visual_type,
        describe_chart,
        extract_data_points,
    ],
    system_prompt="""You are a Visual Content Specialist.
    1. Classify visual content (chart, graph, diagram)
    2. Generate detailed text descriptions
    3. Extract key data points and trends"""
)
```

## LangGraph RAG Workflow

The agentic RAG pipeline uses LangGraph for stateful execution with self-correction:

```python
from langgraph.graph import StateGraph, END

class RAGState(TypedDict):
    query: str
    enhanced_query: str
    documents: List[Document]
    answer: str
    is_relevant: bool
    is_valid: bool
    retry_count: int

# Build the graph
workflow = StateGraph(RAGState)
workflow.add_node("enhance", enhance_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("validate_retrieval", validate_retrieval)
workflow.add_node("generate", generate_answer)
workflow.add_node("validate_answer", validate_answer)

workflow.set_entry_point("enhance")
workflow.add_edge("enhance", "retrieve")
workflow.add_edge("retrieve", "validate_retrieval")
workflow.add_conditional_edges(
    "validate_retrieval",
    lambda x: "generate" if x["is_relevant"] else "enhance"
)
workflow.add_edge("generate", "validate_answer")
workflow.add_conditional_edges(
    "validate_answer",
    lambda x: END if x["is_valid"] else "retrieve"
)

rag_agent = workflow.compile()
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Query Enhancement** | Acronym expansion and semantic rewriting |
| **Hybrid Retrieval** | Dense embeddings + BM25 sparse search with RRF fusion |
| **Cross-encoder Reranking** | Improves precision with semantic relevance scoring |
| **Retrieval Validation** | Ensures retrieved documents are relevant |
| **Answer Validation** | Verifies answers are grounded in source documents |
| **Self-correction Loops** | Automatic retries when validation fails |

## Test Results

The RAG system was tested with 6 real AI research papers (262 pages, 1,131 chunks):

| Document | Source | Pages | Chunks |
|----------|--------|-------|--------|
| Attention Is All You Need | arXiv:1706.03762 | 15 | 60 |
| RAG: Retrieval-Augmented Generation | arXiv:2005.11401 | 19 | 92 |
| ReAct: Reasoning and Acting | arXiv:2210.03629 | 33 | 152 |
| Chain-of-Thought Prompting | arXiv:2201.11903 | 43 | 187 |
| GPT-4 Technical Report | arXiv:2303.08774 | 100 | 422 |
| Generative AI for Dummies | Wiley/Snowflake | 52 | 218 |

### Test Scenarios (21/21 Passed ✅)

| Scenario | Tests | Result |
|----------|-------|--------|
| Basic Factual Retrieval | 3 | ✅ Pass |
| Multi-hop Reasoning | 3 | ✅ Pass |
| Comparison Questions | 3 | ✅ Pass |
| Code/Technical Content | 3 | ✅ Pass |
| Figure/Diagram Understanding | 2 | ✅ Pass |
| Edge Cases & Negatives | 4 | ✅ Pass |
| Complex Multi-part Questions | 3 | ✅ Pass |

### Key Capabilities Verified
- ✅ Cross-document synthesis
- ✅ No hallucinations on unknown topics
- ✅ Exact number/formula retrieval
- ✅ Corrects false premises in questions
- ✅ Vision processing for diagrams/charts

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LangChain** | Core Framework | Document loaders, chains, prompts, embeddings |
| **LangGraph** | Agentic RAG | Stateful workflow with self-correction |
| **OpenAI GPT-5.2** | LLM + Vision | Answer generation + figure analysis (400K context) |
| **OpenAI Embeddings** | text-embedding-3-large | 3072-dimension semantic vectors |
| **Qdrant** | Vector Database | Semantic search with cosine similarity |
| **Celery + Redis** | Task Queue | Async document processing |
| **FastAPI** | API Server | REST endpoints |
| **PyMuPDF** | PDF Processing | Text & image extraction |
| **React + TypeScript** | Frontend | Modern UI |

## License

MIT License
