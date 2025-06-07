import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import torch
import os
import shutil
from sklearn.cluster import KMeans


def read_embeddings_from_files(embedding_dir):
    """
    Read embeddings from JSON files in the specified directory.

    Args:
        embedding_dir (str): Path to the directory containing embedding JSON files.

    Returns:
        list: A list of embeddings.
        list: A list of corresponding file paths.
    """
    embeddings = []
    file_paths = []

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
            file_paths.append(file_path)

    return embeddings, file_paths

def save_clusters(kmeans, file_paths, output_dir):
    """
    Save clusters in separate subdirectories along with centroid files.

    Args:
        kmeans (KMeans): Trained KMeans model.
        file_paths (list): List of file paths corresponding to embeddings.
        output_dir (str): Directory to store the cluster output.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cluster_idx in range(kmeans.n_clusters):
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_idx}")
        os.makedirs(cluster_dir, exist_ok=True)


        centroid_path = os.path.join(cluster_dir, "centroid.json")
        with open(centroid_path, 'w', encoding='utf-8') as f:
            json.dump({"centroid": kmeans.cluster_centers_[cluster_idx].tolist()}, f, ensure_ascii=False, indent=4)


    for i, file_path in enumerate(file_paths):
        cluster_idx = kmeans.labels_[i]
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_idx}")
        shutil.copy(file_path, cluster_dir)

def main():
    embedding_dir = "C:\\Users\\Parsa\\Desktop\\IR_project\\extra_point_IR\\phase_1_output"
    output_dir = "phase_3_output"
    num_clusters = 5


    embeddings, file_paths = read_embeddings_from_files(embedding_dir)
    embeddings = np.array(embeddings)


    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)


    save_clusters(kmeans, file_paths, output_dir)

    print(f"Clustering completed. Results saved in '{output_dir}'.")

if __name__ == "__main__":
    main()
