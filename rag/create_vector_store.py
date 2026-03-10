import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load fraud patterns
with open("rag/fraud_cases.txt", "r") as f:
    fraud_cases = [line.strip() for line in f.readlines()]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert fraud cases into embeddings
embeddings = model.encode(fraud_cases)

embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# Save index
faiss.write_index(index, "rag/fraud_index.faiss")

print("FAISS vector database created successfully!")
print("Number of fraud patterns:", len(fraud_cases))