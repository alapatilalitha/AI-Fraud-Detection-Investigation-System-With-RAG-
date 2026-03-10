import joblib
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load fraud model
model = joblib.load("fraud_model.pkl")

# Fraud knowledge base
fraud_cases = [
    "Large transactions from a new device after password change may indicate account takeover fraud.",
    "Multiple transactions across different geographic locations within minutes can signal card cloning.",
    "High-value electronics purchases are common in credit card fraud schemes.",
    "Transactions occurring at unusual hours with high amounts may indicate automated fraud.",
    "Rapid sequence of transactions across merchants can indicate stolen card testing."
]

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embed_model.encode(fraud_cases)
embeddings = np.array(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Example transaction features (mock input)
transaction_features = np.random.rand(1, 30)

fraud_prob = model.predict_proba(transaction_features)[0][1]

# Transaction description
transaction_description = "large transaction from new device"

query_embedding = embed_model.encode([transaction_description])

distances, indices = index.search(np.array(query_embedding), k=2)

print("\nFraud Detection Result\n")

print("Fraud Probability:", round(fraud_prob, 2))

if fraud_prob > 0.5:

    print("\nAI Investigation Report:\n")

    for i in indices[0]:
        print("-", fraud_cases[i])

else:
    print("\nTransaction appears normal. No fraud patterns detected.")