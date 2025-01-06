# Install necessary libraries
# pip install langchain langchain-openai sentence-transformers faiss-cpu

# Step 1: Import required libraries
from langchain.schema import HumanMessage  # Defines the structure of user messages
from langchain_openai import ChatOpenAI  # Import OpenAI's language model interface
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from sentence_transformers import SentenceTransformer  # Library for generating text embeddings
import faiss  # A library for efficient similarity search
import numpy as np  
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key stored in the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 2: Load a sentence transformer for embedding generation
# Sentence transformers are used to convert textual data into high-dimensional vector representations.
model_name = "sentence-transformers/all-mpnet-base-v2"  # A high-performing transformer model for embeddings
embedder = SentenceTransformer(model_name)  # Load the model

# Step 3: Prepare your knowledge base
# The knowledge base consists of documents representing prior information.
documents = [
    "AI is transforming industries by automating tasks.",
    "RAG combines retrieval and generation for better responses.",
    "LangChain simplifies the creation of intelligent workflows.",
    "FAISS is a library for efficient similarity search.",
    "GPT models are powerful for natural language generation."
]

# Step 4: Generate embeddings for documents
# Each document is converted into a high-dimensional vector for semantic similarity computations.
doc_embeddings = embedder.encode(documents, convert_to_tensor=True).cpu().detach().numpy()

# Step 5: Build a FAISS index
# FAISS (Facebook AI Similarity Search) is a library for fast nearest-neighbor search in large datasets.
dimension = doc_embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(dimension)  # Use L2 norm for similarity
index.add(doc_embeddings)  # Add document embeddings to the index

# Step 6: Define the language model
# Use OpenAI's GPT model for query refinement and response generation.
# "gpt-3.5-turbo" is a highly efficient and cost-effective chat model.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Low temperature for deterministic outputs

# Step 7: Define prompts for query refinement and generation
# Prompts guide the language model's behavior and context understanding.

# Query refinement prompt: Helps rephrase or improve the query for better retrieval.
refine_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an assistant specializing in refining queries for better document retrieval."
    ),
    HumanMessagePromptTemplate.from_template(
        "Current Query: {query}\nContext: {context}\nRefined Query:"
    )
])

# Answer generation prompt: Guides the model to generate a concise, context-based response.
generation_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert in AI and NLP. Generate a concise response based on the context."
    ),
    HumanMessagePromptTemplate.from_template(
        "Context: {context}\nAnswer:"
    )
])

# Step 8: Define the Agentic RAG function
# Agentic RAG combines retrieval and generation capabilities with iterative refinement.

def agentic_rag(query, index, documents, refine_prompt, generation_prompt, llm, k=3):
    """
    Implements an Agentic RAG pipeline for intelligent query refinement and response generation.
    
    Parameters:
        query (str): Initial user query.
        index (faiss.Index): Prebuilt FAISS index of document embeddings.
        documents (list): List of textual documents in the knowledge base.
        refine_prompt (ChatPromptTemplate): Prompt template for query refinement.
        generation_prompt (ChatPromptTemplate): Prompt template for response generation.
        llm (ChatOpenAI): The language model for refinement and generation.
        k (int): Number of documents to retrieve.
    
    Returns:
        dict: Contains the refined query, retrieved contexts, final context, and the generated answer.
    """
    # Step 8.1: Compute the query embedding
    query_embedding = embedder.encode([query], convert_to_tensor=True).cpu().detach().numpy()
    
    # Step 8.2: Retrieve top-k documents from the index
    distances, indices = index.search(query_embedding, k)  # Nearest neighbors search
    retrieved_contexts = [documents[i] for i in indices[0]]  # Extract documents by indices
    
    # Combine retrieved contexts into a single string
    context = " ".join(retrieved_contexts)
    
    # Step 8.3: Refine the query using the language model
    # Use the LLM to improve or rephrase the query for better retrieval.
    refined_query = llm.invoke(refine_prompt.format_messages(query=query, context=context)).content
    
    # Step 8.4: Retrieve refined context
    refined_query_embedding = embedder.encode([refined_query], convert_to_tensor=True).cpu().detach().numpy()
    distances, indices = index.search(refined_query_embedding, k)
    refined_contexts = [documents[i] for i in indices[0]]
    final_context = " ".join(refined_contexts)
    
    # Step 8.5: Generate the final answer using the refined context
    answer = llm.invoke(generation_prompt.format_messages(context=final_context)).content
    
    return {
        "initial_query": query,
        "refined_query": refined_query,
        "retrieved_contexts": retrieved_contexts,
        "final_context": final_context,
        "answer": answer
    }

# Step 9: Test the Agentic RAG system
query = "How is RAG useful in AI workflows?"  # Example user query
result = agentic_rag(query, index, documents, refine_prompt, generation_prompt, llm)

# Display results
print("Initial Query:", result["initial_query"])
print("Refined Query:", result["refined_query"])
print("Retrieved Contexts:", result["retrieved_contexts"])
print("Final Context:", result["final_context"])
print("Generated Answer:", result["answer"])
