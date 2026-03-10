from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_fraud_cases():

    with open("rag/fraud_cases.txt", "r") as f:
        text = f.read()

    cases = text.split("Case")
    cases = [c.strip() for c in cases if c.strip()]

    return cases


def retrieve_fraud_context(query):

    cases = load_fraud_cases()

    embeddings = model.encode(cases)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    query_vector = model.encode([query])

    distances, indices = index.search(query_vector, k=2)

    results = [cases[i] for i in indices[0]]

    return "\n\n".join(results)