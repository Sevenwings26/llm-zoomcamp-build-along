import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import text_format
from openai import OpenAI


# inititalize open ai 
MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"

client = OpenAI(
    base_url='http://localhost:8000/v1',
    api_key='not-needed'
)

def llm(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': text_format},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content


question_prompt = """
question: what are the services that sevevwings offers?
answer: 
"""

answer = llm(question_prompt)

print(answer)
