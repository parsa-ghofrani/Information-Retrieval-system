import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
import os
import time

model_path = "HooshvareLab/bert-fa-zwnj-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return embedding

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def get_top_similar_docs(query_embedding, cluster_dir, k=5):
    cluster_similarities = []

    for cluster_name in os.listdir(cluster_dir):
        cluster_path = os.path.join(cluster_dir, cluster_name)
        if os.path.isdir(cluster_path):
            centroid_path = os.path.join(cluster_path, "centroid.json")

            if os.path.exists(centroid_path):
                with open(centroid_path, 'r', encoding='utf-8') as f:
                    centroid_data = json.load(f)
                    centroid_embedding = np.array(centroid_data["centroid"])

                similarity_to_centroid = cosine_similarity(query_embedding, centroid_embedding)

                cluster_similarities.append((cluster_name, similarity_to_centroid, cluster_path))

    cluster_similarities.sort(key=lambda x: x[1], reverse=True)

    best_cluster_name, best_similarity, best_cluster_path = cluster_similarities[0]

    top_similar_docs = []
    cluster_files = [f for f in os.listdir(best_cluster_path) if f.endswith(".json")]

    for file_name in cluster_files:
        file_path = os.path.join(best_cluster_path, file_name)

        if file_name == "centroid.json":
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                doc_data = json.load(f)

                if isinstance(doc_data, list):
                    doc_data = doc_data[0]

                doc_embedding = doc_data["embedding"]
            except KeyError:
                continue

            similarity_to_doc = cosine_similarity(query_embedding, doc_embedding)
            doc_content = doc_data.get("content", "No content")
            top_similar_docs.append((file_name, similarity_to_doc, doc_content, best_cluster_name))

    top_similar_docs.sort(key=lambda x: x[1], reverse=True)
    return top_similar_docs[:k]

def main():
    cluster_dir = "phase_3_output"
    query = input("Enter the query: ")

    start_time = time.time()

    query_embedding = get_embedding(query)
    k = 5
    top_similar_docs = get_top_similar_docs(query_embedding, cluster_dir, k)

    elapsed_time = time.time() - start_time

    print(f"Time taken to retrieve top {k} documents: {elapsed_time:.4f} seconds\n")
    print("Top similar documents:")
    for rank, (file_name, score, content, cluster) in enumerate(top_similar_docs, start=1):
        print(f"{rank}. Document ID: {file_name}")
        print(f"   Similarity Score: {score:.4f}")
        print(f"   Cluster: {cluster}")
        print(f"   Content: {content[:200]}...")
        print()

if __name__ == "__main__":
    main()
