import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier()

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model/fraud_model.pkl", "wb"))

print("Fraud detection model saved successfully!")