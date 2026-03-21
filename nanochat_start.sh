#!/bin/bash
# Master "Nanochat (RAG/WEB) Start" command
# Ollama backend, RAG enabled, Precision: bf16
export NANOCHAT_DTYPE=bf16
python -m scripts.chat_web --backend ollama --rag --rag-corpus ./data/rag_corpus.txt
