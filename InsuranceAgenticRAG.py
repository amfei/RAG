# Step 1: Import Necessary Libraries
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 2: Simulate Insurance Data
# Example dataset with claims and their fraud status
insurance_data = [
    {"claim_id": "C001", "description": "Car accident with minor damage", "fraud": False},
    {"claim_id": "C002", "description": "House fire claim with suspicious receipts", "fraud": True},
    {"claim_id": "C003", "description": "Theft of expensive jewelry", "fraud": False},
    {"claim_id": "C004", "description": "Flood damage claim with inconsistent reports", "fraud": True},
]

descriptions = [item["description"] for item in insurance_data]
fraud_labels = [item["fraud"] for item in insurance_data]

# Step 3: Create FAISS Index for Retrieval
vectorizer = TfidfVectorizer()
description_embeddings = vectorizer.fit_transform(descriptions).toarray()
index = faiss.IndexFlatL2(description_embeddings.shape[1])
index.add(np.array(description_embeddings).astype('float32'))

# Step 4: Define LLM and Prompts
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Refinement Prompt
refine_prompt = ChatPromptTemplate.from_template(
    """You are an expert insurance analyst. Refine the query to focus on fraud detection and risk analysis.
    Query: {query}
    Context: {context}
    Refined Query:"""
)

# Insight Generation Prompt
generate_prompt = ChatPromptTemplate.from_template(
    """You are an expert insurance advisor. Given the context, provide actionable insights about fraud detection:
    Context: {context}
    Query: {query}
    Insights:"""
)

# Step 5: Define RAG Workflow
def agentic_rag(query, index, insurance_data, refine_chain, generation_chain, k=3):
    # Step 1: Retrieve similar cases
    query_vector = vectorizer.transform([query]).toarray().astype('float32')
    _, neighbors = index.search(query_vector, k)
    retrieved_cases = [insurance_data[i] for i in neighbors[0]]

    # Step 2: Refine the query
    context = "\n".join([case["description"] for case in retrieved_cases])
    refined_query = refine_chain.run(query=query, context=context)

    # Step 3: Generate insights
    insights = generation_chain.run(query=refined_query, context=context)
    return {"refined_query": refined_query, "retrieved_cases": retrieved_cases, "insights": insights}

# Step 6: Create Chains
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)
generation_chain = LLMChain(llm=llm, prompt=generate_prompt)

# Step 7: Test with an Insurance Query
query = "This claim seems unusual. Should I investigate further?"
result = agentic_rag(query, index, insurance_data, refine_chain, generation_chain)

# Step 8: Display Results
print("Original Query:", query)
print("Refined Query:", result["refined_query"])
print("\nRetrieved Cases:")
for case in result["retrieved_cases"]:
    print(case)
print("\nActionable Insights:", result["insights"])
