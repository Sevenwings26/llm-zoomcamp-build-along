import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import json_format
import time
from sqlitesearch import TextSearchIndex


documents = json_format
# print(len(documents))

docs_company = [doc for doc in documents if doc["category"] == "company"]
# print(f"Company: {len(docs_company)} documents")


sqlite_index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["category"],
    db_path="company_faq.db"
)


for doc in docs_company:
    sqlite_index.add(doc)
    print(f"""Added: {doc["question"][:60]}...""")
    time.sleep(0.5)

# sqlite_index.close()
# print("Done. Index saved to company_faq.db")
