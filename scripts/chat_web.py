#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

Launch examples:

- Default: chat via Ollama (no local LLM weights)
python -m scripts.chat_web
# Requires: ollama serve && ollama pull llama3.2  (or set OLLAMA_CHAT_MODEL)

- Ollama + RAG (embeddings + corpus)
python -m scripts.chat_web --rag --rag-corpus dev/rag_sample_corpus.txt

- Local nanochat checkpoints (GPU)
python -m scripts.chat_web --backend nanochat --model-tag d12

- 4 GPUs (nanochat backend only)
python -m scripts.chat_web --backend nanochat --num-gpus 4

To chat, open the URL printed in the console. (If on cloud box, make sure to use public IP)

Endpoints:
  GET  /           - Chat UI
  POST /chat/completions - Chat API (streaming only)
  GET  /health     - Health check with worker pool status
  GET  /stats      - Worker pool statistics and GPU utilization

Persistence (optional):
  Set DATABASE_URL or pass --database-url with an async SQLAlchemy URL, e.g.
  sqlite+aiosqlite:///./data/nanochat.db or postgresql+asyncpg://user:pass@host/db
  Tables: sessions, analytics_events, user_training_text. Without DATABASE_URL,
  analytics and save_to_training use JSONL as before.

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 0-200 (0 disables top-k filtering, using full vocabulary)
  - Max tokens clamped to 1-4096
"""

import argparse
import datetime
import json
import os
import re
import asyncio
import logging
import random
import urllib.error
import urllib.request
import time

import numpy as np
import httpx
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass

import nanochat
logger.info(f"DEBUG: nanochat imported from {nanochat.__file__}")

from nanochat.common import compute_init, autodetect_device_type
from nanochat.netutil import assert_tcp_port_available
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.db.models import Base
from nanochat.db.repository import (
    ensure_chat_session,
    insert_feedback_event,
    insert_metric_event,
    insert_training_text,
)
from nanochat.db.session import create_async_engine_from_url, make_session_factory

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0 # 0 disables top-k filtering, using full vocabulary
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat Web Server')
parser.add_argument('--backend', type=str, default='ollama', choices=['ollama', 'nanochat'], help='ollama: use Ollama HTTP API for chat; nanochat: load local checkpoints')
parser.add_argument('--ollama-chat-model', type=str, default=os.environ.get('OLLAMA_CHAT_MODEL', 'llama3.2'), help='Ollama model name when --backend ollama')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (nanochat backend only; default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl (nanochat backend only)")
parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=100, help='Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load (nanochat backend)')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load (nanochat backend)')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
parser.add_argument('--rag', action='store_true', help='Enable RAG (Ollama embeddings + Chroma or text corpus)')
parser.add_argument('--rag-db', type=str, default='../chroma_db_basecamp', help='Chroma persist dir, or folder with corpus.txt')
parser.add_argument('--rag-corpus', type=str, default=None, help='Plain-text file to chunk and embed via Ollama')
parser.add_argument('--rag-chroma-api-key', type=str, default=None, help='Chroma Cloud API key (or set CHROMA_API_KEY)')
parser.add_argument('--rag-chroma-tenant', type=str, default=None, help='Chroma Cloud tenant (or set CHROMA_TENANT)')
parser.add_argument('--rag-chroma-database', type=str, default=None, help='Chroma Cloud database (or set CHROMA_DATABASE)')
parser.add_argument('--rag-chroma-collection', type=str, default=None, help='Chroma Cloud collection name (default: first collection)')
parser.add_argument('--ollama-url', type=str, default='http://127.0.0.1:11434', help='Ollama API base URL')
parser.add_argument('--ollama-keep-alive', type=str, default='5m', help='Ollama keep_alive parameter (e.g. 5m, 1h, 0)')
parser.add_argument('--ollama-num-ctx', type=int, default=4096, help='Ollama num_ctx parameter')
parser.add_argument('--rag-embed-model', type=str, default='nomic-embed-text', help='Ollama embedding model name')
parser.add_argument('--rag-chunk-size', type=int, default=800, help='Max characters per chunk (corpus mode)')
parser.add_argument('--rag-k', type=int, default=5, help='Number of documents to retrieve for RAG (Target: 90% retrieval accuracy)')
parser.add_argument(
    '--database-url',
    type=str,
    default=None,
    help='SQLAlchemy async URL (sets DATABASE_URL) for analytics + training tables; e.g. sqlite+aiosqlite:///./data/nanochat.db',
)
args = parser.parse_args()
if args.database_url:
    os.environ["DATABASE_URL"] = args.database_url

# Configure logging for conversation traffic
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# RAG System Prompt
RAG_SYSTEM_PROMPT = """You are a highly accurate assistant with web browsing capabilities. Use the following context snippets retrieved from our Chroma DB to answer the user's request. 

GUIDELINES:
1. ONLY use the provided context to answer if possible. 
2. If the context is insufficient or irrelevant, explicitly state: "Based on the retrieved documents, I do not have enough information to answer this accurately." and suggest that they can contribute more information using the /save_to_training endpoint.
3. You can also use your browser tool to find more information. To use the browser, think about your plan inside <|thought_start|> and <|thought_end|>, then emit an action inside <|action_start|> and <|action_end|>.
4. Maintain a professional and concise tone.
5. Do not mention the context unless it helps the user understand the answer.

### RETRIEVED CONTEXT:
{context}"""

if not torch.distributed.is_initialized():
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddp_rank = torch.distributed.get_rank()
    ddp_world_size = torch.distributed.get_world_size()


async def _ollama_embedding(prompt: str, model: str, base_url: str, timeout: float = 120.0) -> list:
    url = base_url.rstrip("/") + "/api/embeddings"
    body = {"model": model, "prompt": prompt}
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            out = resp.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama unreachable at {base_url} ({e}). Run ollama serve and ollama pull {model}") from e
    return out["embedding"]


def _ollama_tags_reachable(base_url: str, timeout: float = 2.0) -> bool:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


async def ollama_chat_stream(
    messages: list,
    temperature: float,
    max_new_tokens: int,
    top_k: int,
) -> AsyncGenerator[str, None]:
    """Stream chat tokens from Ollama /api/chat (NDJSON)."""
    url = args.ollama_url.rstrip("/") + "/api/chat"
    opts = {
        "temperature": temperature,
        "num_predict": max_new_tokens,
        "num_ctx": args.ollama_num_ctx,
    }
    if top_k > 0:
        opts["top_k"] = top_k
    body = {
        "model": args.ollama_chat_model,
        "messages": messages,
        "stream": True,
        "options": opts,
        "keep_alive": args.ollama_keep_alive,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        async with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    yield f"data: {json.dumps({'token': piece, 'gpu': 0}, ensure_ascii=False)}\n\n"
                if obj.get("done"):
                    break
    yield f"data: {json.dumps({'done': True})}\n\n"


def _chunk_corpus(text: str, max_chars: int, overlap: int) -> list:
    text = text.strip()
    if not text:
        return []
    # Improved chunking: try to split by paragraphs, then sentences if still too large
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [text]
    
    chunks = []
    for p in paras:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            # Simple recursive-ish split: if paragraph is too long, split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', p)
            current_chunk = ""
            for s in sentences:
                if len(current_chunk) + len(s) + 1 <= max_chars:
                    current_chunk = (current_chunk + " " + s).strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = s
            if current_chunk:
                chunks.append(current_chunk)
    
    # Final pass to ensure no chunk is over max_chars (fallback to hard split)
    final_chunks = []
    for c in chunks:
        if len(c) <= max_chars:
            final_chunks.append(c)
        else:
            step = max(1, max_chars - overlap)
            for i in range(0, len(c), step):
                final_chunks.append(c[i : i + max_chars])
    return final_chunks


class ChromaOllamaRAG:
    def __init__(self, collection, model: str, base_url: str):
        self._col = collection
        self._model = model
        self._base = base_url

    async def similarity_search(self, query: str, k: int = 3):
        emb = await _ollama_embedding(query, self._model, self._base)
        try:
            n = self._col.count()
        except Exception:
            n = 999999
        n = min(k, max(1, n))
        r = self._col.query(query_embeddings=[emb], n_results=n)
        docs = (r.get("documents") or [[]])[0]
        out = []
        for d in docs:
            o = type("Doc", (), {})()
            o.page_content = d
            out.append(o)
        return out


class OllamaCorpusRAG:
    def __init__(self, chunks: list, embs: np.ndarray, model: str, base_url: str):
        self._chunks = chunks
        self._embs = embs # np.ndarray of shape (num_chunks, dim)
        self._model = model
        self._base = base_url

    @classmethod
    async def create(cls, chunks: list, model: str, base_url: str, batch_size: int = 10):
        print(f"RAG: embedding {len(chunks)} corpus chunks via Ollama ({model}) in parallel...")
        embs = []
        # Parallelize embedding calls in batches to avoid overwhelming Ollama
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            tasks = [_ollama_embedding(ch, model, base_url) for ch in batch]
            batch_embs = await asyncio.gather(*tasks)
            embs.extend(batch_embs)
            print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)}")
        
        print("RAG: corpus index ready.")
        return cls(chunks, np.array(embs), model, base_url)

    async def similarity_search(self, query: str, k: int = 3):
        qe = np.array(await _ollama_embedding(query, self._model, self._base))
        # Vectorized cosine similarity: (A . B) / (||A|| * ||B||)
        # Assuming embeddings might not be normalized
        dot = np.dot(self._embs, qe)
        norm_embs = np.linalg.norm(self._embs, axis=1)
        norm_qe = np.linalg.norm(qe)
        similarities = dot / (norm_embs * norm_qe + 1e-12)
        
        top_k_indices = np.argsort(-similarities)[:k]
        out = []
        for idx in top_k_indices:
            o = type("Doc", (), {})()
            o.page_content = self._chunks[idx]
            out.append(o)
        return out


@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object

class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type) # e.g. cpu|mps
                print(f"Loading model on {device_type}...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    session_id: Optional[str] = None
    ab_group: Optional[str] = None

class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int
    feedback: str # "thumb_up", "thumb_down"
    ab_group: str
    model: str

class MetricRequest(BaseModel):
    session_id: str
    metric_name: str
    metric_value: float
    ab_group: str
    model: str

class SaveToTrainingRequest(BaseModel):
    text: str

def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse."""
    # Check number of messages
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
        )

    # Check individual message lengths and total conversation length
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed"
        )

    # Validate role values
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant", "system"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user', 'assistant', or 'system'"
            )

    # Validate temperature
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
            )

    # Validate top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
            )

    # Validate max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Ollama-backed chat (default) or local nanochat checkpoints."""
    app.state.backend = args.backend
    app.state.ollama_chat_model = args.ollama_chat_model
    app.state.worker_pool = None

    if args.backend == "nanochat":
        print("Loading nanochat models across GPUs...")
        app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
        await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    else:
        print(f"Chat backend: Ollama — model={args.ollama_chat_model} at {args.ollama_url}")
        if not _ollama_tags_reachable(args.ollama_url):
            print(
                "Warning: Ollama does not appear reachable at /api/tags. "
                "Run `ollama serve` and `ollama pull " + args.ollama_chat_model + "`."
            )

    app.state.rag_db = None
    if args.rag:
        script_dir = os.path.dirname(__file__)
        repo_root = os.path.dirname(script_dir)
        db_path = os.path.abspath(os.path.join(script_dir, args.rag_db))
        chroma_sqlite = os.path.join(db_path, "chroma.sqlite3")
        corpus_path = args.rag_corpus
        if not corpus_path:
            for cand in ("corpus.txt", "corpus.md", "rag_corpus.txt"):
                p = os.path.join(db_path, cand)
                if os.path.isfile(p):
                    corpus_path = p
                    break
        if not corpus_path:
            sample = os.path.join(repo_root, "dev", "rag_sample_corpus.txt")
            if os.path.isfile(sample):
                corpus_path = sample
                print(f"RAG: using sample corpus {sample} (override with --rag-corpus)")
        try:
            api_key = args.rag_chroma_api_key or os.environ.get("CHROMA_API_KEY")
            tenant = args.rag_chroma_tenant or os.environ.get("CHROMA_TENANT")
            database = args.rag_chroma_database or os.environ.get("CHROMA_DATABASE")
            if api_key and tenant and database:
                try:
                    import chromadb
                    client = chromadb.CloudClient(
                        api_key=api_key,
                        tenant=tenant,
                        database=database,
                    )
                    cols = client.list_collections()
                    if cols:
                        col_name = args.rag_chroma_collection or cols[0].name
                        col = client.get_collection(name=col_name)
                        n = col.count()
                        if n > 0:
                            app.state.rag_db = ChromaOllamaRAG(col, args.rag_embed_model, args.ollama_url)
                            print(f"RAG: Chroma Cloud ({col_name}, {n} docs) + Ollama {args.rag_embed_model}")
                        else:
                            print("RAG: Chroma Cloud collection empty; trying corpus if available.")
                    else:
                        print("RAG: Chroma Cloud database has no collections; trying corpus if available.")
                except ImportError:
                    print(
                        "RAG: Chroma Cloud requested but chromadb not installed. "
                        "Use Python 3.11+ and `uv sync --extra rag`, or use --rag-corpus."
                    )
            if app.state.rag_db is None and os.path.isfile(chroma_sqlite):
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=db_path)
                    cols = client.list_collections()
                    if cols:
                        col = client.get_collection(cols[0].name)
                        if col.count() > 0:
                            app.state.rag_db = ChromaOllamaRAG(col, args.rag_embed_model, args.ollama_url)
                            print(f"RAG: Chroma ({col.count()} docs) + Ollama {args.rag_embed_model}")
                        else:
                            print("RAG: Chroma collection empty; trying corpus if available.")
                    else:
                        print("RAG: no Chroma collections; trying corpus if available.")
                except ImportError:
                    print(
                        "RAG: chroma.sqlite3 present but chromadb not installed. "
                        "Use Python 3.11+ and `uv sync --extra rag`, or use --rag-corpus."
                    )
            if app.state.rag_db is None and corpus_path and os.path.isfile(corpus_path):
                with open(corpus_path, encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                chunks = _chunk_corpus(raw, args.rag_chunk_size, max(100, args.rag_chunk_size // 4))
                if chunks:
                    app.state.rag_db = await OllamaCorpusRAG.create(chunks, args.rag_embed_model, args.ollama_url)
            if app.state.rag_db is None:
                print(
                    "RAG: disabled. Start Ollama, run `ollama pull nomic-embed-text`, "
                    "then use --rag-corpus FILE, Chroma Cloud (CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE), or a Chroma DB directory."
                )
        except Exception as e:
            print(f"RAG Warning: {e}")

    app.state.db_engine = None
    app.state.db_session_factory = None
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        db_engine = create_async_engine_from_url(db_url)
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        app.state.db_engine = db_engine
        app.state.db_session_factory = make_session_factory(db_engine)
        print("Database: persistence enabled (sessions, analytics_events, user_training_text).")

    print(
        "Inference/RAG startup finished; Uvicorn will print the listen URL when the socket is bound."
    )
    yield

    eng = getattr(app.state, "db_engine", None)
    if eng is not None:
        await eng.dispose()
        app.state.db_engine = None
        app.state.db_session_factory = None

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _dual_write_user_jsonl() -> bool:
    v = os.environ.get("NANOCHAT_DUAL_WRITE_USER_DATA", "1").lower()
    return v not in ("0", "false", "no")


@app.post("/save_to_training")
async def save_to_training(request: SaveToTrainingRequest):
    """Save user-provided text to the DB (if DATABASE_URL) and/or user_data.jsonl."""
    user_data_path = "user_data.jsonl"
    factory = getattr(app.state, "db_session_factory", None)
    if factory:
        try:
            async with factory() as db:
                await insert_training_text(db, request.text, source="save_to_training")
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to save training text to database: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    if not factory or _dual_write_user_jsonl():
        try:
            with open(user_data_path, "a", encoding="utf-8") as f:
                json_line = json.dumps({"text": request.text}, ensure_ascii=False)
                f.write(json_line + "\n")
            logger.info(f"Saved to {user_data_path}: {request.text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to save to {user_data_path}: {e}")
            if not factory:
                raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "message": "Saved to training data"}

@app.post("/log_feedback")
async def log_feedback(request: FeedbackRequest):
    """Log user feedback (thumb up/down)."""
    factory = getattr(app.state, "db_session_factory", None)
    backend = getattr(app.state, "backend", "ollama")
    if factory:
        try:
            async with factory() as db:
                await ensure_chat_session(
                    db,
                    session_id=request.session_id,
                    ab_group=request.ab_group,
                    model=request.model,
                    backend=backend,
                )
                await insert_feedback_event(
                    db,
                    session_id=request.session_id,
                    message_index=request.message_index,
                    feedback=request.feedback,
                    ab_group=request.ab_group,
                    model=request.model,
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        analytics_path = "analytics.jsonl"
        try:
            with open(analytics_path, "a", encoding="utf-8") as f:
                log_entry = {
                    "event": "feedback",
                    "session_id": request.session_id,
                    "message_index": request.message_index,
                    "feedback": request.feedback,
                    "ab_group": request.ab_group,
                    "model": request.model,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    logger.info(f"Logged feedback: {request.feedback} for session {request.session_id}")
    return {"status": "ok"}

@app.post("/log_metric")
async def log_metric(request: MetricRequest):
    """Log performance metrics."""
    factory = getattr(app.state, "db_session_factory", None)
    backend = getattr(app.state, "backend", "ollama")
    if factory:
        try:
            async with factory() as db:
                await ensure_chat_session(
                    db,
                    session_id=request.session_id,
                    ab_group=request.ab_group,
                    model=request.model,
                    backend=backend,
                )
                await insert_metric_event(
                    db,
                    session_id=request.session_id,
                    metric_name=request.metric_name,
                    metric_value=request.metric_value,
                    ab_group=request.ab_group,
                    model=request.model,
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        analytics_path = "analytics.jsonl"
        try:
            with open(analytics_path, "a", encoding="utf-8") as f:
                log_entry = {
                    "event": "metric",
                    "session_id": request.session_id,
                    "metric_name": request.metric_name,
                    "metric_value": request.metric_value,
                    "ab_group": request.ab_group,
                    "model": request.model,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}

@app.get("/")
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Replace the API_URL to use the actual host and port the server is running on
    # In a local setup, window.location.origin is usually sufficient.
    html_content = html_content.replace(
        "const API_URL = '';",
        "const API_URL = window.location.origin;"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """Serve the NanoChat logo for favicon and header."""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """Generate assistant response with streaming."""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    try:
        python_start = worker.tokenizer.encode_special("<|python_start|>")
        python_end = worker.tokenizer.encode_special("<|python_end|>")
        output_start = worker.tokenizer.encode_special("<|output_start|>")
        output_end = worker.tokenizer.encode_special("<|output_end|>")
    except Exception:
        python_start = python_end = output_start = output_end = -1
    stop_on_tool = {assistant_end, bos, python_start, python_end, output_start, output_end} - {-1}

    # Accumulate tokens to properly handle multi-byte UTF-8 characters (like emojis)
    accumulated_tokens = []
    # Track the last complete UTF-8 string (without replacement characters)
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1)
    ):
        token = token_column[0]
        
        # Safety check: if model produces invalid tokens
        if not (0 <= token < worker.tokenizer.get_vocab_size()):
            logger.warning(f"Model produced invalid token {token}, stopping generation.")
            break

        if token in stop_on_tool:
            break

        # Append the token to sequence
        accumulated_tokens.append(token)
        # Decode all accumulated tokens to get proper UTF-8 handling
        # Note that decode is a quite efficient operation, basically table lookup and string concat
        current_text = worker.tokenizer.decode(accumulated_tokens)
        # Only emit text if it doesn't end with a replacement character
        # This ensures we don't emit incomplete UTF-8 sequences
        if not current_text.endswith('�'):
            # Extract only the new text since last clean decode
            new_text = current_text[len(last_clean_text):]
            if new_text:  # Only yield if there's new content
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"


def _append_analytics_jsonl(log_entry: dict) -> None:
    analytics_path = "analytics.jsonl"
    try:
        with open(analytics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except OSError as e:
        logger.error(f"Failed to append analytics: {e}")


async def _persist_generation_metrics(
    session_id: str,
    ab_group: str,
    model: str,
    backend: str,
    metrics: list[tuple[str, float]],
) -> None:
    factory = getattr(app.state, "db_session_factory", None)
    if factory:
        try:
            async with factory() as db:
                await ensure_chat_session(
                    db,
                    session_id=session_id,
                    ab_group=ab_group,
                    model=model,
                    backend=backend,
                )
                for name, val in metrics:
                    await insert_metric_event(
                        db,
                        session_id=session_id,
                        metric_name=name,
                        metric_value=float(val),
                        ab_group=ab_group,
                        model=model,
                    )
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to persist metrics to database: {e}")
    else:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for name, val in metrics:
            _append_analytics_jsonl(
                {
                    "event": "metric",
                    "session_id": session_id,
                    "metric_name": name,
                    "metric_value": val,
                    "ab_group": ab_group,
                    "model": model,
                    "timestamp": ts,
                }
            )


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint (streaming only) — Ollama or local nanochat."""

    validate_chat_request(request)

    # A/B testing logic: assign a group if not provided
    ab_group = request.ab_group or random.choice(["A", "B"])
    session_id = request.session_id or f"sess_{random.randint(0, 1000000)}"

    factory = getattr(app.state, "db_session_factory", None)
    backend_name = "ollama" if app.state.backend == "ollama" else "nanochat"
    model_for_session = args.ollama_chat_model if backend_name == "ollama" else (args.model_tag or "nanochat")
    if factory:
        try:
            async with factory() as db:
                await ensure_chat_session(
                    db,
                    session_id=session_id,
                    ab_group=ab_group,
                    model=model_for_session,
                    backend=backend_name,
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to upsert chat session: {e}")

    logger.info(f"Session: {session_id}, AB Group: {ab_group}")
    logger.info("=" * 20)
    for message in request.messages:
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-" * 20)

    # Example A/B test: Group B gets a more concise system prompt
    modified_messages = list(request.messages)
    if ab_group == "B":
        # Check if there's a system message, if not add one, if yes append to it
        has_system = False
        for msg in modified_messages:
            if msg.role == "system":
                msg.content += "\nAlways be extremely concise and answer in 3 sentences or less."
                has_system = True
                break
        if not has_system:
            modified_messages.insert(0, ChatMessage(role="system", content="Always be extremely concise and answer in 3 sentences or less."))

    rag_context = ""
    if app.state.rag_db is not None and len(modified_messages) > 0:
        last_user_message = next((m.content for m in reversed(modified_messages) if m.role == "user"), "")
        if last_user_message:
            print(f"RAG: Retrieving context for query: '{last_user_message[:50]}...'")
            results = await app.state.rag_db.similarity_search(last_user_message, k=args.rag_k)
            if results:
                rag_context = "Relevant context:\n"
                for doc in results:
                    rag_context += f"- {doc.page_content}\n"
                print(f"RAG: Found {len(results)} context snippets.")

    start_time = time.time()

    if app.state.backend == "ollama":
        ollama_messages = []
        if rag_context:
            ollama_messages.append(
                {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=rag_context)}
            )
        for message in modified_messages:
            if message.role in ("user", "assistant", "system"):
                ollama_messages.append({"role": message.role, "content": message.content})

        temp = request.temperature if request.temperature is not None else args.temperature
        max_nt = request.max_tokens if request.max_tokens is not None else args.max_tokens
        tk = request.top_k if request.top_k is not None else args.top_k

        response_tokens = []

        async def stream_ollama():
            token_count = 0
            try:
                # Send the session/AB info in the first chunk
                yield f"data: {json.dumps({'session_id': session_id, 'ab_group': ab_group, 'model': args.ollama_chat_model})}\n\n"
                async for chunk in ollama_chat_stream(ollama_messages, temp, max_nt, tk):
                    line = chunk.replace("data: ", "").strip()
                    if line:
                        try:
                            chunk_data = json.loads(line)
                            if "token" in chunk_data:
                                response_tokens.append(chunk_data["token"])
                                token_count += 1
                        except json.JSONDecodeError:
                            pass
                    yield chunk
            finally:
                duration = time.time() - start_time
                tps = token_count / duration if duration > 0 else 0.0
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (Ollama {args.ollama_chat_model}): {full_response}")
                logger.info(f"Stats: {token_count} tokens, {duration:.2f}s, {tps:.2f} tok/s")
                logger.info("=" * 20)

                await _persist_generation_metrics(
                    session_id,
                    ab_group,
                    args.ollama_chat_model,
                    "ollama",
                    [
                        ("tokens_per_second", tps),
                        ("total_tokens", float(token_count)),
                        ("generation_time", duration),
                    ],
                )

        return StreamingResponse(stream_ollama(), media_type="text/event-stream")

    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # Build conversation tokens
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]

        # Prepend RAG context if found
        if rag_context:
            conversation_tokens.append(user_start)
            conversation_tokens.extend(worker.tokenizer.encode(RAG_SYSTEM_PROMPT.format(context=rag_context)))
            conversation_tokens.append(user_end)
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(worker.tokenizer.encode("Understood. I will use the provided context from Chroma DB."))
            conversation_tokens.append(assistant_end)

        system_parts = []
        for message in modified_messages:
            if message.role == "system":
                system_parts.append(message.content)
            elif message.role == "user":
                user_content = message.content
                if system_parts:
                    user_content = "\n\n".join(system_parts) + "\n\n" + user_content
                    system_parts = []
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(user_content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        if system_parts:
            conversation_tokens.append(user_start)
            conversation_tokens.extend(worker.tokenizer.encode("\n\n".join(system_parts)))
            conversation_tokens.append(user_end)

        conversation_tokens.append(assistant_start)

        # Streaming response with worker release after completion
        response_tokens = []
        model_name = args.model_tag or "nanochat"
        async def stream_and_release():
            token_count = 0
            try:
                # Send the session/AB info in the first chunk
                yield f"data: {json.dumps({'session_id': session_id, 'ab_group': ab_group, 'model': model_name})}\n\n"
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # Accumulate response for logging
                    line = chunk.replace("data: ", "").strip()
                    if line:
                        try:
                            chunk_data = json.loads(line)
                            if "token" in chunk_data:
                                response_tokens.append(chunk_data["token"])
                                token_count += 1
                        except json.JSONDecodeError:
                            pass
                    yield chunk
            finally:
                duration = time.time() - start_time
                tps = token_count / duration if duration > 0 else 0.0
                # Log the assistant response to console
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info(f"Stats: {token_count} tokens, {duration:.2f}s, {tps:.2f} tok/s")
                logger.info("=" * 20)

                await _persist_generation_metrics(
                    session_id,
                    ab_group,
                    model_name,
                    "nanochat",
                    [
                        ("tokens_per_second", tps),
                        ("total_tokens", float(token_count)),
                        ("generation_time", duration),
                    ],
                )

                # Release worker back to pool after streaming is done
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # Make sure to release worker even on error
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """Health check endpoint."""
    db_on = getattr(app.state, "db_session_factory", None) is not None
    if getattr(app.state, "backend", "nanochat") == "ollama":
        reach = _ollama_tags_reachable(args.ollama_url)
        return {
            "status": "ok",
            "backend": "ollama",
            "ollama_url": args.ollama_url,
            "ollama_chat_model": args.ollama_chat_model,
            "ready": reach,
            "ollama_reachable": reach,
            "database": db_on,
        }
    worker_pool = getattr(app.state, "worker_pool", None)
    return {
        "status": "ok",
        "backend": "nanochat",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0,
        "database": db_on,
    }

@app.get("/stats")
async def stats():
    """Worker pool (nanochat) or Ollama backend info."""
    if getattr(app.state, "backend", "nanochat") == "ollama":
        return {
            "backend": "ollama",
            "ollama_chat_model": args.ollama_chat_model,
            "ollama_url": args.ollama_url,
        }
    worker_pool = app.state.worker_pool
    return {
        "backend": "nanochat",
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [{"gpu_id": w.gpu_id, "device": str(w.device)} for w in worker_pool.workers],
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting NanoChat Web Server")
    print(f"Backend: {args.backend}")
    if args.backend == "ollama":
        print(f"Ollama chat model: {args.ollama_chat_model} @ {args.ollama_url}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    assert_tcp_port_available(args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
