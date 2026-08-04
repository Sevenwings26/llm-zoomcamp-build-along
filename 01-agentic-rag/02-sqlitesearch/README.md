# 02-sqlitesearch: Persistent RAG Search

This folder demonstrates how to implement **Retrieval-Augmented Generation (RAG)** using a **persistent, disk-based database** (SQLite) instead of an in-memory solution. 

By using `sqlitesearch`, we create a permanent search index (`company_faq.db`) on disk. This means that once the data is indexed, it stays there forever (or until deleted) and can be queried instantly across multiple script executions without needing to be rebuilt from scratch every time.

---

## Project Structure & Learning Guide

This project is explicitly broken down into three files to separate concerns—a best practice in software engineering:

### 1. `utils.py` (The Data Pipeline & Indexer)
- **Role**: This script is responsible for the **Data Ingestion** phase.
- **What it does**: 
  - It loads the raw data from our central `data.py` file.
  - It filters the documents (e.g., getting only `"company"` category documents).
  - It initializes the `TextSearchIndex` with our SQLite database path (`company_faq.db`).
  - It iterates over the documents and `.add()`s them to the database.
- **Learning Note**: Because this script performs the insertion, running it rebuilds or adds to the database. In a production environment, you would typically separate the "Indexing Script" from the "Querying Script" so you don't accidentally insert duplicate records every time you start your app.

### 2. `rag.py` (The RAG Engine)
- **Role**: This encapsulates the core logic of the RAG pipeline into a reusable `RAGBase` class.
- **What it does**: 
  - Takes a user query and searches the SQLite database (`search`).
  - Formats the retrieved documents into a text block (`build_context`).
  - Merges the context with the user query using a prompt template (`build_prompt`).
  - Sends the combined prompt to the local LLM (`llm`).
- **Learning Note**: Using a class (`RAGBase`) is a great Object-Oriented Programming (OOP) approach. It makes the system modular, meaning you could easily swap out the SQLite index for a different database (like Pinecone) without changing the core LLM logic!

### 3. `main.py` (The Application Entry Point)
- **Role**: The user-facing script that ties everything together.
- **What it does**: 
  - Initializes the OpenAI client to connect to the local LLM.
  - Instantiates the `RAGBase` assistant using the `sqlite_index` from `utils`.
  - Enters an interactive prompt (`input(">>> ")`) so the user can ask questions in real-time.
- **Learning Note**: Notice how clean this file is! By abstracting the heavy lifting into `utils.py` and `rag.py`, the `main.py` file is incredibly easy to read. This is exactly how production applications are structured.

---

## Why Persistent Search? (vs. In-Memory)

When comparing this to `01-minsearch`, the main takeaway is **Persistence**.
1. **In-Memory (`01-minsearch`)**: Lives in RAM. Super fast to set up, but if your server crashes or your script stops, your entire database is gone. You have to re-read your JSON and rebuild the index every single time.
2. **Persistent (`02-sqlitesearch`)**: Lives on Disk (HDD/SSD). You index your data once. Even if you turn off your computer, the `.db` file remains, and you can query it instantly the next day. This is how real-world applications handle data!

*Designed as a learning exercise for building RAG applications.*
