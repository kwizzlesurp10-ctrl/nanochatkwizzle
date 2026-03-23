#!/bin/bash
export PYTHONPATH=$(pwd)
echo "STARTING EXTREME SERVER"
python3 -u scripts/chat_web.py --backend nanochat --model-tag browser_verify --source sft --port 8011 --device-type cpu
