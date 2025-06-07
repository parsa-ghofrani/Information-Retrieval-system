import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
import os
import time  # For measuring time

# Load the ParsBERT model
model_path = "HooshvareLab/bert-fa-zwnj-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)


def get_embedding(text):
    """Generate embedding for a given text using ParsBERT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the [CLS] token embedding (the first token's representation)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return embedding


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)  # Dot product of a and b
    norm_a = np.linalg.norm(a)  # L2 norm (magnitude) of vector a
    norm_b = np.linalg.norm(b)  # L2 norm (magnitude) of vector b
    if norm_a == 0 or norm_b == 0:  # Check if either vector is zero to avoid division by zero
        return 0.0
    return dot_product / (norm_a * norm_b)


def get_top_k_similar_docs(query, k, embedding_dir):
    """
    Return top k documents most similar to the query based on cosine similarity of embeddings.

    Args:
        query (str): The query text.
        k (int): Number of top similar documents to return.
        embedding_dir (str): Path to the directory containing document embeddings (JSON files).

    Returns:
        list: A list of tuples with document filenames and their similarity scores, sorted by similarity.
    """
    # Generate embedding for the query
    query_embedding = get_embedding(query)

    # Store similarities
    similarities = []

    # Iterate over document embeddings one by one in the specified directory
    for file_name in os.listdir(embedding_dir):
        file_path = os.path.join(embedding_dir, file_name)

        # Ensure it's a JSON file
        if file_name.endswith(".json"):
            # Load document embedding from JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)

            # Handle cases where doc_data is a list or dictionary
            if isinstance(doc_data, list):
                # If the data is a list (containing a single dictionary with embedding)
                doc_embedding = doc_data[0]["embedding"]
                doc_content = doc_data[0].get("content", "No content")
            else:
                # If the data is a single dictionary
                doc_embedding = doc_data["embedding"]
                doc_content = doc_data.get("content", "No content")

            # Compute similarity
            similarity = cosine_similarity(query_embedding, doc_embedding)

            # Store result as (file_name, similarity, content)
            similarities.append((file_name, similarity, doc_content))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k results
    return similarities[:k]


# Example usage
query = input("enter the query: ")
k = 5
embedding_dir = "C:\\Users\\Parsa\\Desktop\\IR_project\\extra_point_IR\\phase_1_output"

# Measure the time it takes from getting the query to showing results
start_time = time.time()

top_k_docs = get_top_k_similar_docs(query, k, embedding_dir)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the results
print(f"Time taken to retrieve top {k} documents: {elapsed_time:.4f} seconds\n")
print("Top similar documents:")
for rank, (file_name, score, content) in enumerate(top_k_docs, start=1):
    print(f"{rank}. File: {file_name} - Similarity: {score:.4f}")
    print(f"   Content: {content[:200]}...")  # Show the first 200 characters of the content
    print()
