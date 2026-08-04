# Module 01: Introduction to Agentic RAG

Welcome to **Module 1**, the starting point for learning Retrieval-Augmented Generation (RAG) and Agentic patterns! This module is designed to help you understand the foundational building blocks of how Large Language Models (LLMs) interact with external data sources to generate accurate, context-aware responses.

## Core Learning Objective
The primary goal here is to deeply understand the **Retrieval** phase of RAG. We explore two fundamentally different approaches to building a search index, utilizing two key libraries:

### 1. `minsearch` (In-Memory Retrieval)
Explored in the `01-minsearch` directory, this approach uses the `minsearch` library.
- **Concept**: A pure-Python, lightweight search engine that runs entirely in your computer's RAM using TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity.
- **Why it matters**: It is the absolute fastest way to prototype. It requires zero database setup, making it an excellent educational tool for grasping how text vectorization and context retrieval work mathematically under the hood.
- **Trade-off**: Data is lost when the script stops running. You must re-index your data every time, which isn't suitable for production but is perfect for learning.

### 2. `sqlitesearch` (Persistent Retrieval)
Explored in the `02-sqlitesearch` directory, this approach introduces the `sqlitesearch` library.
- **Concept**: A disk-based search engine that leverages SQLite to create a permanent database file (e.g., `.db`).
- **Why it matters**: This takes you one step closer to real-world, production-level applications. You index the data once, and it remains permanently saved on disk. Future script executions query the existing database instantly without the overhead of rebuilding the index.
- **Trade-off**: Requires a bit more setup to manage database states, but solves the scalability problem of in-memory systems.

## How to Navigate This Module
If you are just starting out, we recommend reviewing the codebase in order:
1. **Start with `01-minsearch`** to see the bare-bones mechanics of a RAG pipeline and how the prompt is constructed.
2. **Move on to `02-sqlitesearch`** to see how that exact same logic is adapted to use a persistent, stateful database.

---
*Built as part of the LLM Zoomcamp learning journey.*