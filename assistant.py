# assistant.py

import os
import time
import threading
# threading lets us run the file watcher
# AND the chat loop at the same time!
# without it, one would block the other

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
# Observer watches the folder for changes

from watchdog.events import FileSystemEventHandler
# FileSystemEventHandler defines WHAT to do
# when a change is detected


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
    # get just the filename from full path
    # example: "./notes/work.txt" → "work.txt"
    filename = os.path.basename(filepath)

    # only process .txt files
    if not filename.endswith(".txt"):
        return

    # open and read the file
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # split into non-empty lines
    chunks = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    added = 0
    updated = 0

    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}_{i}"

        # convert chunk to embedding
        embedding = embedder.encode(chunk).tolist()

        # check if this chunk ID already exists
        existing = collection.get(ids=[chunk_id])

        if existing["ids"]:
            # chunk exists — check if text changed
            existing_text = existing["documents"][0]

            if existing_text == chunk:
                # exact same text → skip, no change
                continue
            else:
                # text changed! update it
                # ChromaDB uses .update() to replace
                collection.update(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=[embedding]
                )
                updated += 1
        else:
            # brand new chunk → add it
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
        # reprint "You: " so chat doesnt look broken
        # flush=True forces it to print immediately


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
    # This class defines what happens when
    # watchdog detects a change in the notes/ folder

    def on_modified(self, event):
        # called when an existing file is edited and saved
        if not event.is_directory:
            # event.src_path = full path of changed file
            print(f"\n[Detected change] {event.src_path}")
            embed_file(event.src_path)

    def on_created(self, event):
        # called when a NEW file is added to notes/
        if not event.is_directory:
            print(f"\n[New file detected] {event.src_path}")
            # small delay to make sure file is fully written
            # before we try to read it
            time.sleep(0.5)
            embed_file(event.src_path)

    def on_deleted(self, event):
        # called when a file is deleted
        # we just notify — we don't remove from DB
        # (old notes stay searchable even if file deleted)
        if not event.is_directory:
            filename = os.path.basename(event.src_path)
            print(f"\n[File deleted] {filename} "
                  f"(existing chunks kept in DB)")
            print("You: ", end="", flush=True)


def start_watcher():
    # create the watcher and observer
    event_handler = NotesWatcher()
    observer = Observer()

    # tell observer to watch notes/ folder
    # recursive=False means don't watch subfolders
    observer.schedule(event_handler, NOTES_FOLDER, recursive=False)

    # start watching in background
    observer.start()
    print("Watching notes/ folder for changes...")

    return observer
    # we return observer so we can stop it later on quit


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
Keep your answer short and clear.
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

# load all existing notes first
print("Loading your notes...")
load_all_notes()

# start file watcher in background
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
            # stop the background watcher cleanly
            observer.join()
            # wait for it to fully stop
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
    # if user presses Ctrl+C instead of typing quit
    print("\nStopping...")
    observer.stop()
    observer.join()