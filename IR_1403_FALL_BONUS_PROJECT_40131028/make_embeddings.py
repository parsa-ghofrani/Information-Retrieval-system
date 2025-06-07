import json
import torch
from transformers import AutoTokenizer, AutoModel

# Load the ParsBERT model
model_path = "HooshvareLab/bert-fa-zwnj-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)


# Function to get embedding for a given text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()  # [CLS] token embedding
    return embedding


# Load the JSON dataset
input_file = "IR_bonus_dataset.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process each document in the dataset and store each separately
for doc_id, doc_info in data.items():
    content = doc_info["content"]
    embedding = get_embedding(content)
    print(doc_id)

    # Prepare the data structure for the document
    output_data = {
        "doc_id": doc_id,
        "content": content,
        "embedding": embedding
    }

    # Save the result to a new JSON file named after the document ID
    output_file = f"IR_bonus_embedding_{doc_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Embedding for {doc_id} saved to {output_file}")
