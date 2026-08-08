import requests

# Get ingestion file used in learning "01-agentic-rag" 

url = "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/ingest.py"
response = requests.get(url)

if response.status_code == 200:
    with open("ingest.py", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Downloaded ingest.py successfully!")
else:
    print(f"Failed to download. Status code: {response.status_code}")
    