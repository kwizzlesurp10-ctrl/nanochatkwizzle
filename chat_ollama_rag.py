#!/usr/bin/env python3
"""Launch the chat web UI: Ollama for chat (default) + RAG via Ollama embeddings.

Requires:
  - ollama serve
  - ollama pull <OLLAMA_CHAT_MODEL>   (default: llama3.2)
  - ollama pull nomic-embed-text       (for --rag)

Optional args are forwarded to scripts.chat_web (e.g. --port 8080 --rag-corpus path.txt).
"""
import os
import runpy
import sys

if __name__ == "__main__":
    model = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")
    argv_tail = [a for a in sys.argv[1:] if "chat_ollama_rag" not in a]
    sys.argv = [
        "",
        "--backend",
        "ollama",
        "--ollama-chat-model",
        model,
        "--rag",
        *argv_tail,
    ]
    runpy.run_module("scripts.chat_web", run_name="__main__")
