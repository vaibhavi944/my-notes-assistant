# assistant.py

import os
# os lets us read files and folders on your computer

import chromadb
# our vector database — stores and searches embeddings

from groq import Groq
# Groq library — free, super fast AI!

from sentence_transformers import SentenceTransformer
# converts text to embeddings (numbers)
# runs locally on your laptop, totally free!


# ─────────────────────────────────────────────────
# PART 1: SETUP — keys, models, database
# ─────────────────────────────────────────────────

# Your free Groq API key
# paste your key from console.groq.com here
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Connect to Groq
groq_client = Groq(api_key=GROQ_API_KEY)
# this is like logging in to Groq

# Load the embedding model
# runs on YOUR laptop — no API needed!
# downloads once (~90MB), saved locally after that
embedder = SentenceTransformer("all-MiniLM-L6-v2")
# all-MiniLM-L6-v2 = small, fast, accurate
# perfect for personal notes!

# Setup ChromaDB
# PersistentClient means it SAVES to disk
# so next time you run, notes already loaded! ✅
db = chromadb.PersistentClient(path="./chroma_db")

# get_or_create_collection means:
# if "my_notes" exists → use it
# if not → create fresh one
collection = db.get_or_create_collection("my_notes")


# ─────────────────────────────────────────────────
# PART 2: LOAD NOTES INTO VECTOR DB
# ─────────────────────────────────────────────────

def load_notes():
    notes_folder = "./notes"
    # folder where your .txt files live

    total_chunks = 0
    # counter to track how many chunks stored

    # loop through every file in notes/ folder
    for filename in os.listdir(notes_folder):

        # skip anything that isnt a .txt file
        if not filename.endswith(".txt"):
            continue

        # build full path to file
        # example: "./notes/work.txt"
        filepath = os.path.join(notes_folder, filename)

        # open and read the file
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        # encoding="utf-8" handles special characters!

        # split into lines — each line = one chunk
        chunks = [
            line.strip()                  # remove extra spaces
            for line in text.split("\n")  # split by newline
            if line.strip()               # skip empty lines
        ]

        # store each chunk in vector db
        for i, chunk in enumerate(chunks):

            # unique id for each chunk
            # example: "work.txt_0", "ideas.txt_2"
            chunk_id = f"{filename}_{i}"

            # check if chunk already exists in db
            # why? PersistentClient remembers between runs!
            # without this check, duplicates get stored
            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                continue
            # if already exists → skip it

            # convert chunk text to embedding (numbers)
            embedding = embedder.encode(chunk).tolist()
            # .encode()  → text to numpy array
            # .tolist()  → numpy array to plain python list
            # chromadb needs plain list, not numpy!

            # store in chromadb
            collection.add(
                documents=[chunk],      # original text
                embeddings=[embedding], # the numbers
                ids=[chunk_id]          # unique name
            )

            total_chunks += 1

    print(f"Loaded {total_chunks} new chunks into vector DB!")


# ─────────────────────────────────────────────────
# PART 3: SEARCH NOTES
# ─────────────────────────────────────────────────

def search_notes(question):
    # convert question to embedding (numbers)
    question_embedding = embedder.encode(question).tolist()
    # same process as chunks — text → numbers

    # search vector db for top 3 closest chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
        # n_results=3 → give me 3 best matches
        # more context = better answer!
    )

    # results structure looks like:
    # {
    #   "documents": [["chunk1", "chunk2", "chunk3"]],
    #   "ids":       [["work.txt_0", "ideas.txt_1", ...]]
    # }
    # [0] gets results for our first (only) query
    chunks = results["documents"][0]

    return chunks
    # returns list of 3 most relevant text chunks


# ─────────────────────────────────────────────────
# PART 4: ASK GROQ (FREE + FAST LLM!)
# ─────────────────────────────────────────────────

def ask_groq(question, context_chunks):
    # join 3 chunks into one block of context
    context = "\n".join(context_chunks)
    # "\n".join puts each chunk on its own line
    # example output:
    # "The deadline is 15th June.
    #  I want to learn ML in 2025.
    #  Save 2 lakhs by December."

    # build the full prompt
    prompt = f"""
You are a helpful personal assistant.
Use ONLY the notes below to answer the question.
If the answer is not in the notes, say "I don't have notes on that."
Keep your answer short and clear.

Notes:
{context}

Question: {question}

Answer:"""
    # why "use ONLY the notes"?
    # so Groq doesnt make stuff up from its own training!
    # we want answers ONLY from our notes

    # call Groq API
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        # LLaMA 3.3 70B — very powerful free model!
        # runs on Groq's special fast chips
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
        # max length of the answer
        # 500 tokens = roughly 350 words
        # plenty for notes questions!
    )

    # extract just the answer text
    return response.choices[0].message.content
    # .choices[0]          → first response (only one)
    # .message.content     → the actual answer string


# ─────────────────────────────────────────────────
# PART 5: MAIN — run the assistant!
# ─────────────────────────────────────────────────

# load all notes into vector db first
print("Loading your notes...")
load_notes()
# reads all .txt files from notes/ folder
# converts each line to embedding
# stores in chromadb (skips duplicates!)

print("\nNotes assistant ready!")
print("Ask me anything about your notes.")
print("Type 'quit' to exit.\n")

# keep looping until user types quit
while True:

    # wait for user to type something
    question = input("You: ").strip()
    # .strip() removes accidental spaces

    # exit if user types quit
    if question.lower() == "quit":
        print("Bye!")
        break

    # skip if user just pressed enter
    if not question:
        continue

    # STEP 1: find relevant chunks from notes
    print("Searching your notes...")
    relevant_chunks = search_notes(question)

    # STEP 2: send chunks + question to Groq
    print("Thinking...\n")
    answer = ask_groq(question, relevant_chunks)

    # STEP 3: show the answer!
    print(f"Assistant: {answer}\n")