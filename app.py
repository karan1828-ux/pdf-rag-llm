import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.tools import tool
import requests

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (e.g., 2+2, 5*7)."""
    try:
        # Only allow safe characters
        if not all(c in "0123456789+-*/(). " for c in expression):
            return "Invalid characters in expression."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool("get_joke", return_direct=True)
def get_joke(_: str = "") -> str:
    """Fetch a random joke from an external API."""
    try:
        response = requests.get("https://official-joke-api.appspot.com/random_joke")
        if response.status_code == 200:
            data = response.json()
            return f"{data['setup']} {data['punchline']}"
        else:
            return "Failed to fetch a joke."
    except Exception as e:
        return f"Error: {e}"

class PDFRAG:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        # Use local HuggingFace embeddings (no API key needed)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._prepare()

    def _prepare(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
        # Use Llama.cpp for local LLM (no API key needed)
        # Note: You'll need to download a GGUF model file and update the path
        import os
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "models", "llama-2-7b-chat.Q2_K.gguf")
        if not os.path.exists(model_path):
            print(f"Warning: Llama model file not found at {model_path}. Using mock LLM for testing.")
            self.llm = MockLLM()
        else:
            self.llm = LlamaCpp(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4,
                temperature=0.1,
                max_tokens=512,
                verbose=True,
            )
        
        # Use ConversationBufferWindowMemory for conversational memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=3,  # Number of previous exchanges to remember
            return_messages=True
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            memory=self.memory
        )

    def answer_query(self, query):
        return self.qa.run(query)

    def calculate(self, expression):
        return calculator.run(expression)

    def get_joke(self):
        return get_joke.run("")

# Mock LLM class for testing when model file is not available
class MockLLM:
    def __call__(self, prompt):
        return "This is a mock response for testing. The actual Llama model would provide a real answer here."
    
    def invoke(self, prompt):
        return self(prompt)

# Example usage (for testing)
if __name__ == "__main__":
    pdf_path = "test_sample.pdf"  # Replace with your PDF file
    rag = PDFRAG(pdf_path)
    question = "What is this document about?"
    print(rag.answer_query(question)) 
