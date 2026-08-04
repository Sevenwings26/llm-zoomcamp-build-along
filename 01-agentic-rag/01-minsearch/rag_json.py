import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import json_format
from utils import search_index, build_prompt, llm

search_index_list = search_index(json_format)


def rag(query):
    """
    Execute the RAG (Retrieval-Augmented Generation) pipeline.

    This function retrieves relevant context based on the query,
    builds a prompt, and calls the language model to generate an answer.

    Args:
        query (str): The user's question.

    Returns:
        str: The generated answer from the LLM.
    """
    search_results = search_index_list.search(query=query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

query = input(">>> ")
# query = "Tell me about sevenwings"
answer = rag(query)
print(f"\nAnswer : {answer}")
