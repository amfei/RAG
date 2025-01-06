

#!pip install langchain openai faiss-cpu chromadb python-dotenv
#!pip install -U langchain-community
#!pip install unstructured
#!pip install pdf2image pytesseract
#!pip install -U langchain-openai
#!pip install python-dotenv
#!pip install --upgrade openai
#!pip install langchain openai faiss-cpu PyPDF2

# Import necessary libraries
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Create a .env file programmatically for storing sensitive API keys securely
# Note: In Colab, we need to create environment variables like API keys manually in a .env file 
with open(".env", "w") as f:
    f.write("OPENAI_API_KEY=your_openai_api_key_here")  # Replace with actual key if running locally

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key stored in the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 1: Load Documents - Here we load a PDF document (e.g., "NVIDIA.pdf") from the `doc` folder.
pdf_loader = PyPDFLoader("./doc/NVIDIA.pdf")  # Replace with actual document path
documents = pdf_loader.load()

# Step 2: Split Documents - Split the loaded documents into smaller chunks to manage large text sizes.
# We want to split the document into chunks of 1000 characters with 100 characters overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)  # Returns a list of smaller document chunks

# Step 3: Generate Embeddings - Use OpenAI’s pre-trained model to generate embeddings for the document chunks.
embeddings = OpenAIEmbeddings()

# Step 4: Create FAISS Vector Store - Using the embeddings, we create a FAISS index that allows efficient similarity-based search.
# This helps in retrieving relevant chunks of documents based on cosine similarity when querying.
vector_store = FAISS.from_documents(docs, embeddings)

# Step 5: Build the RAG (Retrieval-Augmented Generation) Pipeline
# 5.1 Create a Retriever - The retriever fetches relevant documents based on the query’s similarity with stored documents.
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Fetch top 5 closest documents

# 5.2 Define the Language Model - Here, we initialize the OpenAI language model (e.g., GPT-4).
from langchain_openai.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0, model="gpt-4")  # Choose the GPT-4 model, set temperature to 0 for deterministic output

# 5.3 Combine Retriever and LLM - Create a QA chain that combines the retriever (which gets documents) and the LLM (which generates answers).
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # Use the LLM (GPT-4)
    retriever=retriever,  # Use the retriever that fetches relevant docs
    return_source_documents=True  # Also return the source documents that contributed to the answer
)

# Step 6: Interactive Query Interface
# Now we allow users to ask questions interactively, and the system will use RAG to retrieve relevant documents and generate answers.
while True:
    query = input("Ask your question: ")  # Take user input
    if query.lower() in ["exit", "quit"]:  # Exit the loop if the user types "exit" or "quit"
        print("Exiting RAG application.")
        break

    # Step 7: Get the Answer - Use the RAG pipeline to retrieve relevant documents and generate an answer to the query.
    result = qa_chain({"query": query})

    # Step 8: Display the Answer and the Source Documents
    print("\nAnswer:", result["result"])  # Output the generated answer
    print("\nSource Documents:")  # List the source documents that contributed to the answer
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")  # Display a snippet of each source document
