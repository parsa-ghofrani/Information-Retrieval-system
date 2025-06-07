import json
import os
import numpy as np  # For computing the Euclidean norm

# Path to directory containing document embeddings
embedding_dir = "C:\\Users\\Parsa\\Desktop\\IR_project\\extra_point_IR\\phase_1_output"

# Function to check the Euclidean norm of embeddings
def check_embedding_norms(embedding_dir):
    embedding_norms = {}

    # Iterate over document embeddings one by one
    for file_name in os.listdir(embedding_dir):
        file_path = os.path.join(embedding_dir, file_name)

        # Load document embedding from JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)

            # Assuming embeddings are stored in a list of dictionaries
            if isinstance(doc_data, list):
                # If doc_data is a list, inspect its first element
                doc_embedding = doc_data[0]["embedding"]
            else:
                doc_embedding = doc_data["embedding"]

        # Compute the Euclidean norm (magnitude) of the embedding vector
        embedding_norm = np.linalg.norm(doc_embedding)  # Euclidean norm

        # Print the norm for each document
        print(f"Document: {file_name}, Norm: {embedding_norm:.4f}")

# Run the check
check_embedding_norms(embedding_dir)
