# 01-minsearch: In-Memory RAG Search

This folder demonstrates how to implement a basic **Retrieval-Augmented Generation (RAG)** pipeline using **minsearch**, an in-memory search engine. 

This approach serves as an introduction to how search indexes and RAG context retrieval work under the hood.

---

## How It Works

`minsearch` is a lightweight, pure-Python search engine. 
- **Under the Hood**: It uses TF-IDF (via `scikit-learn`) to vectorize your documents. When you make a query, it converts your query into a vector and uses cosine similarity to find the most relevant documents.
- **Data Types**: It supports text fields (TF-IDF), keyword fields (exact string matching), and numeric/date fields (range filtering) internally using `pandas` DataFrames.
- **In-Memory**: The index is built in your computer's RAM. It does not save anything to your hard drive.

### Project Structure
- **`utils.py`**: Contains the logic to build the search index from the JSON data, construct the prompt with context, and call the local LLM.
- **`rag_json.py`**: The entry point. It loads the data, creates the in-memory index, and processes the user query through the RAG pipeline.

---

## Pros & Cons

**Pros**:
- **Zero Setup**: No need to install, configure, or run a database server (like Elasticsearch or Postgres).
- **Fast Prototyping**: It runs instantly on your local machine, making it perfect for learning and testing small datasets.

**Cons**:
- **No Persistence**: Because it lives in RAM, the index is completely destroyed every time the script finishes. 
- **Scalability**: You must re-index the data from scratch upon every execution, which becomes a bottleneck for larger datasets.

---

## Next Steps: Persistent Search
If you are looking for an approach where the data is permanently saved to disk (so you don't have to rebuild the index every time), please check out the **[02-sqlitesearch](../02-sqlitesearch/README.md)** folder!
