import numpy as np
import json
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def read_embeddings_from_files(embedding_dir):
    embeddings = []
    for file_name in os.listdir(embedding_dir):
        file_path = os.path.join(embedding_dir, file_name)
        if file_name.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
                if isinstance(doc_data, list):
                    doc_embedding = doc_data[0]["embedding"]
                else:
                    doc_embedding = doc_data["embedding"]
                embeddings.append(doc_embedding)
    return np.array(embeddings)



embedding_dir = "C:\\Users\\Parsa\\Desktop\\IR_project\\extra_point_IR\\phase_1_output"

embeddings_matrix = read_embeddings_from_files(embedding_dir)



def print_kmeans_analysis(data, max_k=10):
    print("KMeans Clustering Analysis\n")
    print("{:<8} {:<15}".format("k", "Inertia"))
    print("-" * 38)

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)

        inertia = kmeans.inertia_

        silhouette_avg = silhouette_score(data, kmeans.labels_)

        print("{:<8} {:<15.4f}".format(k, inertia))


print_kmeans_analysis(embeddings_matrix)
