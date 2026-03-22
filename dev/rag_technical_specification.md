# Technical Specification: NanoChat RAG System (Ollama + ChromaDB)

## 1. System Architecture Overview
The NanoChat RAG system provides a "Retrieval-Augmented Generation" pipeline by combining a high-performance vector database (**ChromaDB**) with a local LLM server (**Ollama**). It supports both **Persistent DB Mode** (Chroma Cloud/SQLite) and **On-the-Fly Corpus Mode** (Plain-text indexing).

### Core Components:
- **FastAPI Backend:** Orchestrates the UI, RAG logic, and worker pool.
- **Ollama API:** Handles both embeddings (`/api/embeddings`) and text generation (`/api/chat`).
- **ChromaDB:** Stores and retrieves high-dimensional vector embeddings of document chunks.
- **Chunking Engine:** Splits long documents into manageable, contextually coherent segments using recursive paragraph/sentence splitting.

## 2. API Endpoints (FastAPI)

### `POST /chat/completions` (Streaming)
- **Request Body:** `ChatRequest` (Messages, Temperature, etc.)
- **Logic:**
    1. Extract the last user message.
    2. Query ChromaDB or the memory-indexed corpus for top-k relevant context.
    3. Prepend the context to the conversation tokens.
    4. Stream the generated response from Ollama or local worker.
- **Returns:** NDJSON stream of tokens and session metadata.

### `GET /health`
- **Returns:** Status of Ollama connectivity, worker pool availability, and RAG backend readiness.

### `POST /save_to_training`
- **Request Body:** `SaveToTrainingRequest` (JSONL text)
- **Logic:** Appends user-provided text to `user_data.jsonl` for future fine-tuning.

## 3. Data Schema & Collection Strategy

### Document Chunk Schema (ChromaDB Metadata)
```json
{
  "id": "uuid-v4",
  "embedding": [0.123, -0.456, ...],  // 768 or 1024-dim vector (nomic-embed-text)
  "document": "Plain text chunk content (max ~800 chars)",
  "metadata": {
    "source": "path/to/file.txt",
    "timestamp": "ISO-8601",
    "chunk_id": 42
  }
}
```

### Retrieval Logic:
- **Distance Metric:** Cosine Similarity (Vectorized via NumPy).
- **Strategy:** Top-K retrieval (Default `k=3`, configurable via `--rag-k`).
- **Embedder:** `nomic-embed-text` (Default model in Ollama).

## 4. Performance Metrics & Optimization

### Key Performance Indicators (KPIs):
- **Retrieval Latency (RL):** Time to embed a query and retrieve k documents from ChromaDB. (Goal: < 50ms for k=3).
- **Generation Throughput (Tokens/Sec):** Measured in the `/chat/completions` endpoint. (Goal: > 15 tps on A100, > 5 tps on CPU).
- **Startup Indexing Time:** Time to embed a 1MB text corpus. (Optimized via parallel batch embedding).
- **Context Utilization:** Ratio of retrieved context that is relevant to the query (qualitative).

### Performance Optimizations (Already Implemented):
- **Parallel Embedding:** Batch processing of document chunks via `asyncio.gather` for faster startup.
- **Vectorized Similarity:** Uses NumPy's `np.dot` and `np.linalg.norm` for sub-millisecond similarity calculations.
- **Async API Calls:** Uses `httpx.AsyncClient` for non-blocking communication with the Ollama server.
- **Streaming Response:** NDJSON-based streaming reduces time-to-first-token (TTFT) for better UX.

## 5. Deployment Options
- **Local:** `python -m scripts.chat_web --rag --rag-corpus my_data.txt`
- **Persistent DB:** `python -m scripts.chat_web --rag --rag-db ./chroma_persistent_db`
- **Cloud:** Integrates with Chroma Cloud using `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE`.
