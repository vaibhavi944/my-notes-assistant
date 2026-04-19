# Personal Notes Assistant

A RAG (Retrieval-Augmented Generation) app that lets you ask questions about your own `.txt` notes and get accurate answers. Answers come only from your notes — the LLM does not use its own training data to respond.

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| ChromaDB | latest | Vector database — stores and searches embeddings |
| sentence-transformers | latest | Embedding model (`all-MiniLM-L6-v2`) — runs locally |
| Groq API | latest | LLM API — runs LLaMA 3.3 70B, free tier |

---

## Project Structure

```
my_notes_assistant/
│
├── notes/               ← .txt files with your notes (one fact per line)
│   ├── work.txt
│   ├── ideas.txt
│   └── goals.txt
│
├── chroma_db/           ← auto-created by ChromaDB, stores embeddings on disk
├── assistant.py         ← all RAG logic
└── requirements.txt     ← dependencies
```

---

## How It Works

### Phase 1 — Indexing (runs once)

1. Reads every `.txt` file from the `notes/` folder
2. Splits each file into lines — each line is one chunk
3. Each chunk is converted to an embedding using `all-MiniLM-L6-v2` (384-dimensional vector)
4. Chunk text + embedding + unique ID (`filename_lineindex`) stored in ChromaDB
5. ChromaDB persists to disk at `./chroma_db` — skips already-stored chunks on subsequent runs

### Phase 2 — Querying (every time user asks)

1. User types a question
2. Question is converted to a 384-dimensional embedding using the same model
3. ChromaDB does cosine similarity search — returns top 3 closest chunks
4. Top 3 chunks are joined and inserted into a prompt
5. Prompt sent to Groq API (`llama-3.3-70b-versatile`)
6. Answer returned and printed

---

## Embedding Model

- Model: `all-MiniLM-L6-v2` from `sentence-transformers`
- Dimensions: 384
- Runs fully locally — no API call needed
- Downloads once (~90MB), cached after that
- Chosen over ChromaDB's built-in embeddings for better semantic accuracy

---

## LLM

- Provider: Groq
- Model: `llama-3.3-70b-versatile`
- Free tier: 14,400 requests/day, 6,000 tokens/minute
- `max_tokens` set to 500 per response
- System prompt instructs the model to answer only from provided context

---

## Setup

### 1. Get a free Groq API key

Go to [console.groq.com](https://console.groq.com) → API Keys → Create API Key. No credit card required.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
chromadb
sentence-transformers
groq
```

### 3. Add notes

Create `.txt` files in the `notes/` folder. One fact or sentence per line.

```
notes/work.txt
───────────────────────────────────────
The deadline for the mobile app is 15th June.
We need to hire 2 Flutter developers by next month.
The client wants daily updates on progress.
```

### 4. Add API key

In `assistant.py`:
```python
GROQ_API_KEY = "gsk_..."
```

### 5. Run

```bash
python assistant.py
```

---

## Example Output

```
Loading your notes...
Loaded 12 new chunks into vector DB!

Notes assistant ready!
Ask me anything about your notes.
Type 'quit' to exit.

You: what is the deadline for the mobile app?
Searching your notes...
Thinking...
Assistant: The deadline for the mobile app is 15th June.

You: what are my fitness goals?
Assistant: You want to exercise every morning for at least 30 minutes.

You: what is the weather today?
Assistant: I don't have notes on that.
```

---

## Notes Format

- One sentence or fact per line
- Each line becomes one searchable chunk
- Empty lines are skipped
- Supports UTF-8 (Hindi, special characters work)
- Only `.txt` files are read — other file types in `notes/` are ignored

---

## ChromaDB Behaviour

- Collection name: `my_notes`
- Storage: `./chroma_db/` (persistent, survives restarts)
- Each chunk ID: `{filename}_{line_index}` e.g. `work.txt_0`
- Duplicate check: if ID already exists in DB, chunk is skipped — no duplicates on re-run
- Search: returns top 3 results by cosine similarity

---

## Prompt Template

```
You are a helpful personal assistant.
Use ONLY the notes below to answer the question.
If the answer is not in the notes, say "I don't have notes on that."
Keep your answer short and clear.

Notes:
{top_3_chunks}

Question: {user_question}

Answer:
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `model not found` | Wrong model name | Use `llama-3.3-70b-versatile` |
| `quota exceeded` | Hit rate limit | Wait 1 minute, retry |
| `ModuleNotFoundError` | Missing library | Run `pip install -r requirements.txt` |
| `InvalidAPIKey` | Wrong or missing key | Check key at console.groq.com |
| `no such file or directory` | `notes/` folder missing | Create the folder and add `.txt` files |