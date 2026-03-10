from rag.retriever import retrieve_fraud_context
import ollama


def fraud_investigation(transaction):

    context = retrieve_fraud_context(transaction)

    prompt = f"""
You are a financial fraud investigator.

Transaction:
{transaction}

Related fraud knowledge:
{context}

Provide analysis with:

Fraud Indicators
Fraud Pattern
Explanation
Recommendation
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]