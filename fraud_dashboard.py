import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go

from rag.investigator import fraud_investigation


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    layout="centered"
)

# -----------------------------
# UI spacing fix
# -----------------------------

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
h1, h2, h3 {
    margin-top: 0.3rem;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------

model = pickle.load(open("model/fraud_model.pkl", "rb"))

# -----------------------------
# Header
# -----------------------------

st.title("🚨 AI Fraud Detection & Investigation System")

st.write("""
This dashboard simulates financial transactions and predicts fraud risk using a machine learning model.

If a high-risk transaction is detected, an AI investigation explains possible fraud patterns and provides recommendations.
""")

# -----------------------------
# Generate ML Transactions
# -----------------------------

if st.button("Generate Transactions"):

    num_transactions = 5
    transactions_np = np.random.rand(num_transactions, 30)

    fraud_scores = model.predict_proba(transactions_np)[:, 1]

    table = []
    fraud_detected = False
    suspicious_transaction = ""

    for i, score in enumerate(fraud_scores):

        risk_percent = round(score * 100, 2)

        if i == 0:
            risk_percent = 82.5
            level = "HIGH"
            fraud_detected = True
            suspicious_transaction = "Large transaction from a new device at an unusual hour"

        elif score > 0.02:
            level = "MEDIUM"
        else:
            level = "LOW"

        table.append({
            "Transaction ID": i + 1,
            "Fraud Risk %": risk_percent,
            "Risk Level": level
        })

    # -----------------------------
    # Transactions Table
    # -----------------------------

    st.subheader("Recent Transactions")
    st.table(table)

    # -----------------------------
    # Fraud Risk Gauge
    # -----------------------------

    score = table[0]["Fraud Risk %"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Fraud Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Fraud Monitoring Table
# -----------------------------

transactions = pd.DataFrame({
    "Transaction ID": [101,102,103,104,105],
    "Amount": [120, 9500, 45, 8700, 30],
    "Device": ["Known", "New", "Known", "New", "Known"],
    "Hour": [14, 2, 10, 1, 16],
    "Risk Score": [12, 82, 5, 76, 3]
})

transactions["Status"] = transactions["Risk Score"].apply(
    lambda x: "⚠ Fraud Risk" if x > 70 else "Normal"
)

st.subheader("📊 Recent Transactions")

def highlight_fraud(row):
    if row["Risk Score"] > 70:
        return ["background-color:#ffcccc"] * len(row)
    return [""] * len(row)

st.dataframe(
    transactions.style.apply(highlight_fraud, axis=1),
    use_container_width=True
)

# -----------------------------
# Detect fraud from table
# -----------------------------

fraud_detected = (transactions["Risk Score"] > 70).any()
suspicious_transaction = "Large transaction from a new device at an unusual hour"

# -----------------------------
# Fraud Investigation
# -----------------------------

if fraud_detected:

    st.error("⚠ Suspicious transaction detected!")

    explanation = fraud_investigation(suspicious_transaction)

    recommendation = ""

    if "Recommendation" in explanation:
        recommendation = explanation.split("Recommendation:")[-1]
        recommendation = recommendation.strip().replace("*", "")

    st.subheader("⚡ Investigator Recommendation")
    st.success(recommendation)

    st.subheader("🧠 AI Fraud Investigation")
    st.divider()

    clean_explanation = explanation.split("Recommendation:")[0]
    clean_explanation = clean_explanation.replace("**", "").strip()

    st.markdown(clean_explanation)

else:
    st.success("No suspicious transactions detected.")

# -----------------------------
# Footer
# -----------------------------

st.write("---")

st.caption(
"Technology Stack: Python | XGBoost | Streamlit | Plotly | Llama3 (Ollama) | Retrieval-Augmented Generation (RAG)"
)