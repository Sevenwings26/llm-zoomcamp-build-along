INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

class RAGBase:
    """
    RAGBase encapsulates the entire Retrieval-Augmented Generation pipeline.
    By using a class, we keep our code modular, meaning we could easily swap out 
    the search index or LLM client without changing the rest of our application.
    """
    def __init__(self, index, llm_client, instructions=INSTRUCTIONS, prompt_template=PROMPT_TEMPLATE,
        company_name="sevevwingsINC", model="Qwen/Qwen2.5-3B-Instruct-AWQ"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.company_name = company_name
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        """
        Step 1: Retrieve relevant documents from our persistent SQLite database.
        It uses 'boost_dict' to prioritize certain fields (like the question itself) 
        and 'filter_dict' to ensure we only get results for the correct company.
        """
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"company_name": self.company_name}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        """
        Step 2: Format the retrieved documents into a single text block.
        The LLM needs context in plain text, not Python dictionaries, so we 
        extract the relevant parts and join them together with newlines.
        """
        lines = []

        for doc in search_results:
            lines.append(doc["category"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        """
        Step 3: Combine the user's query and the formatted context into our prompt template.
        This string is what will actually be sent to the LLM to process.
        """
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )
        
    def llm(self, prompt):
        """
        Step 4: Send the final prompt to the Language Model.
        We pass in developer instructions to set the system behavior, and 
        the user prompt containing the context.
        """
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text
    
    def rag(self, query):
        """
        Main orchestration function: Combines all the steps of the RAG pipeline.
        1. Search for context
        2. Build the prompt
        3. Query the LLM
        """
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

        