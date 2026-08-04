# from sqlitesearch import TextSearchIndex

# sqlite_index = TextSearchIndex(
#     text_fields=["question", "section", "answer"],
#     keyword_fields=["category"],
#     db_path="company_faq.db"
# )

# search_results = sqlite_index.search(query="What is sevevwings", num_results=2)
# print(search_results)



from rag import RAGBase
# from utils import 
from openai import OpenAI
from utils import sqlite_index


client = OpenAI(
    base_url="http://localhost:8000/v1/",
    api_key="not-needed"
)

assistant = RAGBase(
    index=sqlite_index,
    llm_client=client
)

user_insert = input(">>> ")
answer = assistant.rag(user_insert)
print(f"\nAnswer : {answer}")


