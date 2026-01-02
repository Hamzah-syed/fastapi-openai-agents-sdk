# FastAPI + OpenAI Agents SDK

A minimal FastAPI server using OpenAI Agents SDK with Google Gemini.

## Setup

```bash
uv sync
```

Create `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_key_here
```

## Run

```bash
uv run main.py
```

Server runs at http://localhost:8000

API docs: http://localhost:8000/docs

## Usage

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 25 * 4}'
```
