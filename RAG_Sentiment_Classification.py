
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import faiss  # For creating a similarity-based retrieval index
import numpy as np
import os 

# Install the necessary dependencies (ensure these packages are available in your environment)
#!pip install transformers datasets torch faiss-cpu

# Hugging Face authentication (optional but recommended)

# Retrieve the token from the environment variables
hf_token = os.getenv("HF_TOKEN")

# Login to Hugging Face
if hf_token:
    login(hf_token)
else:
    raise ValueError("Hugging Face token not found. Please check your .env file.")
# Step 1: Load the dataset and preprocess
# Load a financial sentiment dataset for prediction. It includes sentences and sentiment labels.
dataset = load_dataset("financial_phrasebank", "sentences_allagree", split="train")

# Step 2: Shuffle and sample the dataset for retrieval task
# Use a subset of 500 random sentences from the dataset to build the retrieval system
retrieval_subset = dataset.shuffle(seed=42).select(range(500))  # Use 500 samples for retrieval
sentences = retrieval_subset["sentence"]  # Extract sentences
labels = retrieval_subset["label"]  # Extract sentiment labels

# Step 3: Load pre-trained tokenizer and model
# Load a model for zero-shot classification (we use a pre-trained BART model for this purpose)
model_name = "facebook/bart-large-mnli"  # A model suitable for multi-class classification
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Tokenizer to process text
model = AutoModelForSequenceClassification.from_pretrained(model_name)  # Model for sequence classification

# Step 4: Create an embedding-based index using FAISS
# Function to generate sentence embeddings for retrieval task using the BART model
def get_embeddings(texts, batch_size=32):
    """Generate embeddings for a list of texts using the tokenizer and model."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]  # Process texts in batches
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")  # Tokenize the batch
        with torch.no_grad():  # No gradient calculation required for inference
            outputs = model(**inputs).logits  # Forward pass to get the output logits
            batch_embeddings = outputs.cpu().numpy()  # Convert the logits to NumPy arrays
        all_embeddings.append(batch_embeddings)  # Collect all embeddings
    return np.vstack(all_embeddings)  # Return embeddings as a stacked array

# Generate embeddings for all sentences in the retrieval subset
embeddings = get_embeddings(sentences)

# Step 5: Normalize the embeddings for cosine similarity
# Normalize embeddings to unit vectors to perform cosine similarity later
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Step 6: Create a FAISS index to store embeddings and enable fast similarity search
index = faiss.IndexFlatIP(embeddings.shape[1])  # FAISS index with inner product search (cosine similarity)
index.add(embeddings)  # Add embeddings to the index for retrieval

# Step 7: Define a function for sentiment prediction using RAG (Retrieval-Augmented Generation)
def predict_sentiment_rag(sentence, index, sentences, labels, k=5):
    """Predict sentiment by retrieving similar examples and using majority voting."""
    # Convert the input sentence to its embedding
    input_embedding = get_embeddings([sentence])  # Generate embedding for the input sentence
    input_embedding = input_embedding / np.linalg.norm(input_embedding, axis=1, keepdims=True)  # Normalize the embedding

    # Step 8: Retrieve the k closest sentences based on cosine similarity
    distances, neighbors = index.search(input_embedding, k=k)  # Retrieve k closest embeddings from the index
    neighbor_sentences = [sentences[i] for i in neighbors[0]]  # Get the sentences corresponding to the neighbors
    neighbor_labels = [labels[i] for i in neighbors[0]]  # Get the sentiment labels for those sentences

    # Step 9: Majority voting from the retrieved examples to predict sentiment
    sentiment_prediction = max(set(neighbor_labels), key=neighbor_labels.count)  # Majority vote to decide the sentiment
    return sentiment_prediction, neighbor_sentences  # Return the predicted sentiment and evidence

# Step 10: Test the RAG-based sentiment prediction
test_sentence = "The company's performance exceeded expectations."  # Test sentence for sentiment prediction
prediction, evidence = predict_sentiment_rag(test_sentence, index, sentences, labels)
print(f"Predicted Sentiment: {prediction}")  # Print the predicted sentiment
print(f"Retrieved Sentences: {evidence}")  # Print the retrieved similar sentences as evidence
