#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 -u -m scripts.chat_web --backend nanochat --model-tag browser_verify --source sft --port 8011 --device-type cpu
