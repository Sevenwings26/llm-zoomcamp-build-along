import json
from openai import OpenAI
from sqlitesearch import TextSearchIndex

# 1. Initialize the OpenAI-compatible client
# This client communicates with your local LLM server (e.g. running at localhost:8000)
client = OpenAI(
    base_url='http://localhost:8000/v1',
    api_key='not-needed'
)

MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"

# 2. Define the conversation history.
# We start with the user's question. Under a tool-use/agent architecture, the LLM will 
# analyze this history and decide whether or not it needs to call a tool to answer.
messages = [
    {"role": "user", "content": "I just discovered about Sevenwings. Tell me more?"}
]

# 3. Connect to the persistent SQLite Search Index we built earlier.
# This index reads from 'company_faq.db'.
index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["category"],
    db_path="company_faq.db"
)

# 4. Define the local Python function that executes the search.
# This function queries the SQLite index and returns the raw search results.
def search(query):
    boost_dict = {"question": 3.0, "section": 0.5}
    # We keep filter_dict empty to allow searching across all document categories
    filter_dict = {}

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict
    )

# 5. Define the tool schema (Function Declaration).
# The LLM cannot execute Python code. Instead, we describe our Python function in a 
# JSON-compatible structure so the model knows:
#   - The function's name ('search')
#   - What it does ('Search the FAQ database...')
#   - What arguments it expects ('query' of type string)
search_tool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the FAQ database for entries matching the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the FAQ."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

print("--- Step 1: Sending user query to the LLM with the search tool registered ---")

# 6. Call the LLM. We pass the conversation history AND register our list of tools.
# The LLM will see the user's message, see that it does not know the answer, and notice
# that it has a registered 'search' tool it can call to find out.
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=[search_tool],
)

# Get the assistant's response message
assistant_message = response.choices[0].message

# 7. Check if the LLM decided to call our tool.
# In a standard RAG tool-use pattern:
#   - The LLM does NOT execute the Python code itself. 
#   - It simply returns a request (tool call) telling US: "Hey, please run 'search' with argument 'query' and tell me the result."
if assistant_message.tool_calls:
    print("\n[LLM Decision] The LLM decided it needs to call a tool to answer this query!")
    
    # We append the assistant's tool-call message to the conversation history.
    # The API requires this so it knows what request the tool output is responding to.
    messages.append(assistant_message)
    
    for tool_call in assistant_message.tool_calls:
        print(f"-> LLM requested tool: {tool_call.function.name}")
        print(f"-> LLM requested arguments: {tool_call.function.arguments}")
        
        # 8. Parse the arguments provided by the LLM and execute the local Python function
        args = json.loads(tool_call.function.arguments)
        search_results = search(**args)
        result_json = json.dumps(search_results, indent=2)
        
        print("\n[Local Execution] Ran local search index. Found context:")
        print(result_json)
        
        # 9. Send the tool execution results back to the conversation history.
        # We append a message with the role 'tool', matching the 'tool_call_id'.
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": result_json,
        })

    print("\n--- Step 2: Sending the retrieved search results back to the LLM ---")
    
    # 10. Call the LLM a second time.
    # Now, the conversation history contains:
    #   1. User: "Tell me more about Sevenwings"
    #   2. Assistant: "I need to run search(query='Sevenwings')"
    #   3. Tool: "[Search results showing what Sevenwings is]"
    # The LLM will read the search results and generate the final answer for the user.
    second_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[search_tool],
    )
    
    print("\n[LLM Final Answer]:")
    print(second_response.choices[0].message.content)
else:
    # If the LLM did not call a tool, we print the direct response
    print("\n[LLM Direct Answer]:")
    print(assistant_message.content)



