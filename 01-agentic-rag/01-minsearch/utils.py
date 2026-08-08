from openai import OpenAI
from minsearch import Index

"""
What is minsearch?
    `Minsearch` is a lightweight, pure-Python, in-memory search engine. It was popularized by Alexey Grigorev and the DataTalks.Club community, largely as an educational tool for courses like the LLM Zoomcamp to teach the fundamentals of Retrieval-Augmented Generation (RAG) without the overhead of setting up a heavy database like Elasticsearch or Postgres/pgvector.

Here is a breakdown of how it works under the hood:

### 1. **How It Searches Text**
It is built on top of `scikit-learn`. For text fields, it uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to vectorize your documents. When you query the index, it converts your query into a vector and uses **cosine similarity** to find the documents whose vectors are closest to your query. 

### 2. **Filtering Capabilities**
Even though it's simple, it supports a variety of data types and filtering methods using `pandas` DataFrames internally:
- **Text fields**: (e.g., `question`, `answer`) searched using TF-IDF.
- **Keyword fields**: (e.g., `category` or `course`) searched using exact string matching.
- **Numeric & Date fields**: Allows for range filtering (e.g., finding documents created after a specific date).

### 3. **Why use it?**
- **Zero Setup**: You don't need to spin up a Docker container, configure a vector database, or manage API keys for cloud databases.
- **Perfect for Prototyping**: It's entirely in-memory, meaning it runs instantly on your local machine. It's great for testing small RAG pipelines (like the one you are building in `rag_json.py`) before migrating to a production-ready vector database (like Pinecone, Qdrant, or Elasticsearch).

In your script `rag_json.py`, `minsearch` takes the JSON data, creates a TF-IDF matrix in memory (`search_index`), and then when you send a query, it mathematically finds the most relevant documents in your JSON file to give the Language Model the right context! 
"""

# inititalize open ai 
MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"

client = OpenAI(
    base_url='http://localhost:8000/v1',
    api_key='not-needed'
)

# build prompt 
INSTRUCTIONS = """
Answer questions based on the context provided, do not answer out of the context.
If the answer is not found in the context, respond with 'Question Asked is not related to Sevenwings INC!'
"""
# INSTRUCTIONS = """
# Answer questions based on the context provided, and also use your general knowledge to answer questions that are not in the context.
# Do not hallucinate. If the answer is not found in the context, and general knowledge doesn't correlate, respond with, "Question Asked is not related to Sevenwings INC!"
# """

USER_PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


def search_index(json_data):
    """
    Initialize and fit the search index with provided JSON data.

    Args:
        json_data (list): A list of dictionaries containing the documents to index.

    Returns:
        Index: A fitted minsearch Index object.
    """
    #inititalize the index
    index = Index(
        text_fields=["question", "answer"],
        keyword_fields=["category"]
    )
    index.fit(json_data)
    return index


def build_context(search_results):
    """
    Format search results into a single string to serve as context.

    Args:
        search_results (list): A list of document dictionaries returned from the search index.

    Returns:
        str: A formatted string containing the questions and answers.
    """
    lines = []

    for doc in search_results:
        lines.append(doc["question"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")

    return "\n".join(lines).strip()


def build_prompt(query, search_results):
    """
    Construct the final prompt for the LLM using the user query and search results.

    Args:
        query (str): The user's original question.
        search_results (list): Documents retrieved from the search index.

    Returns:
        str: The full prompt containing the user question and the retrieved context.
    """
    context = build_context(search_results)
    return USER_PROMPT_TEMPLATE.format(question=query, context=context)


def llm(prompt):
    """
    Send the formatted prompt to the language model and get the generated response.

    Args:
        prompt (str): The prompt containing instructions, context, and query.

    Returns:
        str: The generated text response from the language model.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': INSTRUCTIONS},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content
