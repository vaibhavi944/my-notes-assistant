# Personal Notes Assistant

A RAG (Retrieval-Augmented Generation) app that lets you ask questions about your own `.txt` notes and get accurate answers. Answers come only from your notes. Automatically detects and indexes new or edited notes without restarting.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| ChromaDB | Vector database — stores and searches embeddings, persists to disk |
| sentence-transformers (`all-MiniLM-L6-v2`) | Embedding model — runs locally, 384 dimensions |
| Groq API (`llama-3.3-70b-versatile`) | LLM — generates answers, free tier |
| watchdog | File system watcher — detects changes in `notes/` folder in real time |

---

## Project Structure

```
my_notes_assistant/
│
├── notes/               ← .txt files with your notes (one fact per line)
│   ├── work.txt
│   ├── ideas.txt
│   ├── goals.txt
│   ├── fitness.txt
│   ├── learning.txt
│   └── finance.txt
│
├── chroma_db/           ← auto-created, stores embeddings on disk
├── assistant.py         ← all RAG logic
└── requirements.txt     ← dependencies
```

---

## Features

- Ask questions in plain English about your notes
- Answers sourced only from your notes — LLM does not use its own training data
- Auto-detects new `.txt` files added to `notes/` folder while app is running
- Auto-detects edits to existing files and updates embeddings instantly
- No restart needed for any changes
- Duplicate-safe — re-running the app never double-stores chunks
- Persistent vector DB — embeddings saved to disk, survive restarts

---

## How It Works

### Phase 1 — Indexing (startup)

```
Every .txt file in notes/ folder
            ↓
Each line → one chunk
            ↓
all-MiniLM-L6-v2 converts chunk to 384-dim embedding
            ↓
ChromaDB stores: text + embedding + ID (filename_lineindex)
            ↓
If ID already exists and text unchanged → skip
If ID already exists and text changed   → update embedding
If ID is new                            → add embedding
            ↓
Saved to ./chroma_db/ (persists between runs)
```

### Phase 2 — File Watching (runs in background thread)

```
watchdog watches notes/ folder continuously

New file added    → on_created() fires → embed_file() runs
File edited+saved → on_modified() fires → embed_file() runs
File deleted      → on_deleted() fires  → notifies user
                                           (chunks kept in DB)
```

### Phase 3 — Querying (every user question)

```
User types question
            ↓
all-MiniLM-L6-v2 converts question to 384-dim embedding
            ↓
ChromaDB cosine similarity search → top 3 closest chunks
            ↓
top 3 chunks + question inserted into prompt
            ↓
Groq API called (llama-3.3-70b-versatile, max_tokens=500)
            ↓
Answer returned and printed
```

---

## Prompt Template

```
You are a helpful personal assistant.
Use ONLY the notes below to answer the question.
If the answer is not in the notes, say "I don't have notes on that."
Always answer in complete, proper sentences.
The notes are written by the user about themselves.
Always refer to the user as "you" or "your", never use "I" or "my".

Notes:
{top_3_chunks}

Question: {user_question}

Answer:
```

Notes are written in first person ("My trainer is Sanjay") so the prompt explicitly instructs the model to convert to second person ("Your trainer is Sanjay") when answering.

---

## Setup

### 1. Get a free Groq API key

Go to [console.groq.com](https://console.groq.com) → API Keys → Create API Key. No credit card required. Free tier: 14,400 requests/day, 6,000 tokens/minute.

### 2. Install dependencies

```
chromadb
sentence-transformers
groq
watchdog
```

```bash
pip install -r requirements.txt
```

### 3. Add API key

Create a file named `.env` in the root folder and add your key:
```
GROQ_API_KEY=your_key_here
```

### 4. Add your notes

### 5. Run

```bash
python assistant.py
```

---

## Example Output

```
Loading your notes...
Loaded 6 files into vector DB!
Watching notes/ folder for changes...

Notes assistant ready!
Ask me anything about your notes.
Add or edit any .txt file in notes/ — updates instantly!
Type 'quit' to exit.

You: who is my trainer?
Assistant: Your trainer's name is Sanjay.

You: what is my bench press goal?
Assistant: Your bench press goal is to reach 80kg by December.

You: what is the weather today?
Assistant: I don't have notes on that.
```

---

## Auto-Update in Action

While the app is running, add a new line to any `.txt` file and save:

```
The new backend API will be built in FastAPI.
```

Terminal instantly shows:

```
[Detected change] ./notes/work.txt
[Auto-update] work.txt: 1 new chunks added, 0 chunks updated
You:
```

Next question immediately reflects the new info:

```
You: what will the backend be built in?
Assistant: The new backend API will be built in FastAPI.
```

---

## ChromaDB Details

| Property | Value |
|----------|-------|
| Collection name | `my_notes` |
| Storage path | `./chroma_db/` |
| Chunk ID format | `{filename}_{line_index}` e.g. `work.txt_0` |
| Embedding dimensions | 384 |
| Search type | Cosine similarity |
| Results returned | Top 3 per query |
| Duplicate handling | Skip if unchanged, update if changed |

---

## Notes Format Rules

- One sentence or fact per line
- Each line = one searchable chunk
- Empty lines are skipped automatically
- Only `.txt` files are processed — other file types ignored
- UTF-8 encoding supported

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `invalid_api_key` | Wrong or missing Groq key | Check key at console.groq.com, no extra spaces |
| `model not found` | Wrong model name | Use `llama-3.3-70b-versatile` |
| `quota exceeded` | Hit rate limit | Wait 1 minute, retry |
| `ModuleNotFoundError` | Missing library | Run `pip install -r requirements.txt` |
| `no such file or directory` | `notes/` folder missing | Create the folder and add `.txt` files |
| answers say "my" instead of "your" | First person notes | Prompt fix already applied — model uses "you/your" |