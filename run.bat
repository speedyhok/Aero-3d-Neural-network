@echo off
set PATH=C:\Users\Roko\.local\bin;%PATH%
echo Starting AI-Driven Aero Design Accelerator...
echo Open http://127.0.0.1:8000 in your browser.
echo.
uv run uvicorn backend.server_3d:app --reload --host 127.0.0.1 --port 8000
