#!/usr/bin/env bash
echo "Starting GitChatGPT..."
source ./venv/Scripts/activate
docker compose -p gitchatgpt up -d --build