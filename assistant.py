# assistant.py

import os
import time
import threading
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# ─────────────────────────────────────────────────
# PART 1: SETUP
# ─────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOTES_FOLDER = "./notes"

groq_client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("my_notes")


# ─────────────────────────────────────────────────
# PART 2: EMBED AND STORE A SINGLE FILE
# ─────────────────────────────────────────────────

def embed_file(filepath):
    filename = os.path.basename(filepath)

    if not filename.endswith(".txt"):
        return

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    added = 0
    updated = 0

    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}_{i}"
        embedding = embedder.encode(chunk).tolist()
        existing = collection.get(ids=[chunk_id])

        if existing["ids"]:
            existing_text = existing["documents"][0]
            if existing_text == chunk:
                continue
            else:
                collection.update(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=[embedding]
                )
                updated += 1
        else:
            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding]
            )
            added += 1

    if added > 0 or updated > 0:
        print(f"\n[Auto-update] {filename}: "
              f"{added} new chunks added, "
              f"{updated} chunks updated")
        print("You: ", end="", flush=True)


# ─────────────────────────────────────────────────
# PART 3: LOAD ALL NOTES AT STARTUP
# ─────────────────────────────────────────────────

def load_all_notes():
    total = 0
    for filename in os.listdir(NOTES_FOLDER):
        filepath = os.path.join(NOTES_FOLDER, filename)
        if filename.endswith(".txt"):
            embed_file(filepath)
            total += 1
    print(f"Loaded {total} files into vector DB!")


# ─────────────────────────────────────────────────
# PART 4: FILE WATCHER
# ─────────────────────────────────────────────────

class NotesWatcher(FileSystemEventHandler):

    def on_modified(self, event):
        if not event.is_directory:
            print(f"\n[Detected change] {event.src_path}")
            embed_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            print(f"\n[New file detected] {event.src_path}")
            time.sleep(0.5)
            embed_file(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            filename = os.path.basename(event.src_path)
            print(f"\n[File deleted] {filename} "
                  f"(existing chunks kept in DB)")
            print("You: ", end="", flush=True)


def start_watcher():
    event_handler = NotesWatcher()
    observer = Observer()
    observer.schedule(event_handler, NOTES_FOLDER, recursive=False)
    observer.start()
    print("Watching notes/ folder for changes...")
    return observer


# ─────────────────────────────────────────────────
# PART 5: SEARCH NOTES
# ─────────────────────────────────────────────────

def search_notes(question):
    question_embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )
    return results["documents"][0]


# ─────────────────────────────────────────────────
# PART 6: ASK GROQ
# ─────────────────────────────────────────────────

def ask_groq(question, context_chunks):
    context = "\n".join(context_chunks)

    prompt = f"""
You are a helpful personal assistant.
Use ONLY the notes below to answer the question.
If the answer is not in the notes, say "I don't have notes on that."
Always answer in complete, proper sentences.
The notes are written by the user about themselves.
Always refer to the user as "you" or "your", never use "I" or "my".

Notes:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────────
# PART 7: MAIN
# ─────────────────────────────────────────────────

print("Loading your notes...")
load_all_notes()

observer = start_watcher()

print("\nNotes assistant ready!")
print("Ask me anything about your notes.")
print("Add or edit any .txt file in notes/ — updates instantly!")
print("Type 'quit' to exit.\n")

try:
    while True:
        question = input("You: ").strip()

        if question.lower() == "quit":
            print("Stopping file watcher...")
            observer.stop()
            observer.join()
            print("Bye!")
            break

        if not question:
            continue

        print("Searching your notes...")
        relevant_chunks = search_notes(question)

        print("Thinking...\n")
        answer = ask_groq(question, relevant_chunks)

        print(f"Assistant: {answer}\n")

except KeyboardInterrupt:
    print("\nStopping...")
    observer.stop()
    observer.join()