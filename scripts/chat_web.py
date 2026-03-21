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

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 0-200 (0 disables top-k filtering, using full vocabulary)
  - Max tokens clamped to 1-4096
"""

import argparse
import json
import os
import asyncio
import logging
import random
import urllib.error
import urllib.request

import httpx
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass

from nanochat.common import compute_init, autodetect_device_type
from nanochat.netutil import assert_tcp_port_available
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

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
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
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
parser.add_argument('--rag-embed-model', type=str, default='nomic-embed-text', help='Ollama embedding model name')
parser.add_argument('--rag-chunk-size', type=int, default=800, help='Max characters per chunk (corpus mode)')
args = parser.parse_args()

# Configure logging for conversation traffic
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)


def _ollama_embedding(prompt: str, model: str, base_url: str, timeout: float = 120.0) -> list:
    url = base_url.rstrip("/") + "/api/embeddings"
    body = json.dumps({"model": model, "prompt": prompt}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
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
    opts = {"temperature": temperature, "num_predict": max_new_tokens}
    if top_k > 0:
        opts["top_k"] = top_k
    body = {
        "model": args.ollama_chat_model,
        "messages": messages,
        "stream": True,
        "options": opts,
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


def _cosine_vec(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na * nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def _chunk_corpus(text: str, max_chars: int, overlap: int) -> list:
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [text]
    chunks = []
    for p in paras:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            step = max(1, max_chars - overlap)
            for i in range(0, len(p), step):
                chunks.append(p[i : i + max_chars])
    return chunks


class ChromaOllamaRAG:
    def __init__(self, collection, model: str, base_url: str):
        self._col = collection
        self._model = model
        self._base = base_url

    def similarity_search(self, query: str, k: int = 3):
        emb = _ollama_embedding(query, self._model, self._base)
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
    def __init__(self, chunks: list, model: str, base_url: str):
        self._chunks = chunks
        self._model = model
        self._base = base_url
        self._embs = []
        print(f"RAG: embedding {len(chunks)} corpus chunks via Ollama ({model})...")
        for i, ch in enumerate(chunks):
            self._embs.append(_ollama_embedding(ch, model, base_url))
            if (i + 1) % 5 == 0 or i + 1 == len(chunks):
                print(f"  {i + 1}/{len(chunks)}")
        print("RAG: corpus index ready.")

    def similarity_search(self, query: str, k: int = 3):
        qe = _ollama_embedding(query, self._model, self._base)
        ranked = sorted(
            zip(self._embs, self._chunks),
            key=lambda t: -_cosine_vec(qe, t[0]),
        )
        out = []
        for _, ch in ranked[:k]:
            o = type("Doc", (), {})()
            o.page_content = ch
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
                    app.state.rag_db = OllamaCorpusRAG(chunks, args.rag_embed_model, args.ollama_url)
            if app.state.rag_db is None:
                print(
                    "RAG: disabled. Start Ollama, run `ollama pull nomic-embed-text`, "
                    "then use --rag-corpus FILE, Chroma Cloud (CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE), or a Chroma DB directory."
                )
        except Exception as e:
            print(f"RAG Warning: {e}")

    print(
        "Inference/RAG startup finished; Uvicorn will print the listen URL when the socket is bound."
    )
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/save_to_training")
async def save_to_training(request: SaveToTrainingRequest):
    """Save user-provided text to the local training dataset."""
    user_data_path = "user_data.jsonl"
    try:
        with open(user_data_path, "a", encoding="utf-8") as f:
            json_line = json.dumps({"text": request.text}, ensure_ascii=False)
            f.write(json_line + "\n")
        logger.info(f"Saved to {user_data_path}: {request.text[:50]}...")
        return {"status": "ok", "message": "Saved to training data"}
    except Exception as e:
        logger.error(f"Failed to save to {user_data_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_feedback")
async def log_feedback(request: FeedbackRequest):
    """Log user feedback (thumb up/down)."""
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
                "timestamp": torch.utils.benchmark.timer().timeit(1).times[0] if hasattr(torch.utils, 'benchmark') else 0 # Dummy timestamp or use datetime
            }
            # Use datetime instead
            import datetime
            log_entry["timestamp"] = datetime.datetime.now().isoformat()
            f.write(json.dumps(log_entry) + "\n")
        logger.info(f"Logged feedback: {request.feedback} for session {request.session_id}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_metric")
async def log_metric(request: MetricRequest):
    """Log performance metrics."""
    analytics_path = "analytics.jsonl"
    try:
        with open(analytics_path, "a", encoding="utf-8") as f:
            import datetime
            log_entry = {
                "event": "metric",
                "session_id": request.session_id,
                "metric_name": request.metric_name,
                "metric_value": request.metric_value,
                "ab_group": request.ab_group,
                "model": request.model,
                "timestamp": datetime.datetime.now().isoformat()
            }
            f.write(json.dumps(log_entry) + "\n")
        # logger.info(f"Logged metric: {request.metric_name}={request.metric_value} for session {request.session_id}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Failed to log metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Replace the API_URL to use the same origin
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
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

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint (streaming only) — Ollama or local nanochat."""

    validate_chat_request(request)

    # A/B testing logic: assign a group if not provided
    ab_group = request.ab_group or random.choice(["A", "B"])
    session_id = request.session_id or f"sess_{random.randint(0, 1000000)}"

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
            results = app.state.rag_db.similarity_search(last_user_message, k=3)
            if results:
                rag_context = "Relevant context:\n"
                for doc in results:
                    rag_context += f"- {doc.page_content}\n"
                print(f"RAG: Found {len(results)} context snippets.")

    import time
    start_time = time.time()

    if app.state.backend == "ollama":
        ollama_messages = []
        if rag_context:
            ollama_messages.append(
                {"role": "system", "content": "Use the following context when answering the user:\n\n" + rag_context}
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
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (Ollama {args.ollama_chat_model}): {full_response}")
                logger.info(f"Stats: {token_count} tokens, {duration:.2f}s, {token_count/duration:.2f} tok/s")
                logger.info("=" * 20)
                
                # Log to analytics
                log_metric_internal(session_id, "tokens_per_second", token_count/duration if duration > 0 else 0, ab_group, args.ollama_chat_model)
                log_metric_internal(session_id, "total_tokens", token_count, ab_group, args.ollama_chat_model)
                log_metric_internal(session_id, "generation_time", duration, ab_group, args.ollama_chat_model)

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
            conversation_tokens.extend(worker.tokenizer.encode("Use this context to answer the following question: " + rag_context))
            conversation_tokens.append(user_end)
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(worker.tokenizer.encode("Understood. I will use the provided Basecamp context."))
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
                # Log the assistant response to console
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info(f"Stats: {token_count} tokens, {duration:.2f}s, {token_count/duration:.2f} tok/s")
                logger.info("="*20)
                
                # Log to analytics
                log_metric_internal(session_id, "tokens_per_second", token_count/duration if duration > 0 else 0, ab_group, model_name)
                log_metric_internal(session_id, "total_tokens", token_count, ab_group, model_name)
                log_metric_internal(session_id, "generation_time", duration, ab_group, model_name)
                
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

def log_metric_internal(session_id, metric_name, metric_value, ab_group, model):
    """Internal helper to log metrics to file."""
    analytics_path = "analytics.jsonl"
    try:
        with open(analytics_path, "a", encoding="utf-8") as f:
            import datetime
            log_entry = {
                "event": "metric",
                "session_id": session_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "ab_group": ab_group,
                "model": model,
                "timestamp": datetime.datetime.now().isoformat()
            }
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log internal metric: {e}")

@app.get("/health")
async def health():
    """Health check endpoint."""
    if getattr(app.state, "backend", "nanochat") == "ollama":
        reach = _ollama_tags_reachable(args.ollama_url)
        return {
            "status": "ok",
            "backend": "ollama",
            "ollama_url": args.ollama_url,
            "ollama_chat_model": args.ollama_chat_model,
            "ready": reach,
            "ollama_reachable": reach,
        }
    worker_pool = getattr(app.state, "worker_pool", None)
    return {
        "status": "ok",
        "backend": "nanochat",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0,
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
