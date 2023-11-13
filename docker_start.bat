@echo off
echo Starting GitChatGPT Docker...
call .\venv\Scripts\activate.bat
docker compose -p gitchatgpt up -d --build
pause