import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# Settings
# ==========================================

INPUT_FILE = "data/features.csv"

MODEL_FILE = "data/phishing_model.pkl"


# ==========================================
# Load dataset
# ==========================================

print("=" * 70)
print("             PHISHING URL ML TRAINING")
print("=" * 70)

print("\nLoading feature dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# ==========================================
# Separate features and labels
# ==========================================

X = df.drop(columns=["label"])

y = df["label"]


print("\nNumber of features:", X.shape[1])

print("Number of samples:", X.shape[0])


# ==========================================
# Train/Test Split
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# Scale features for Logistic Regression
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# Define models
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# Train and evaluate models
# ==========================================

results = {}

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 70)
    print("Training:", name)
    print("=" * 70)

    # Logistic Regression uses scaled data
    if name == "Logistic Regression":

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

    else:

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)


    # --------------------------------------
    # Calculate metrics
    # --------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    # --------------------------------------
    # Store results
    # --------------------------------------

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    trained_models[name] = model


    # --------------------------------------
    # Display results
    # --------------------------------------

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


    # --------------------------------------
    # Confusion Matrix
    # --------------------------------------

    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")

    print(cm)


# ==========================================
# Display comparison
# ==========================================

print("\n")
print("=" * 70)
print("                 MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':25}"
    f"{'Accuracy':12}"
    f"{'Precision':12}"
    f"{'Recall':12}"
    f"{'F1':12}"
)

print("-" * 70)


for name, metrics in results.items():

    print(
        f"{name:25}"
        f"{metrics['accuracy']:<12.4f}"
        f"{metrics['precision']:<12.4f}"
        f"{metrics['recall']:<12.4f}"
        f"{metrics['f1']:<12.4f}"
    )


# ==========================================
# Select model using F1 score
# ==========================================

best_model_name = max(
    results,
    key=lambda model: results[model]["f1"]
)

best_model = trained_models[best_model_name]


print("\n" + "=" * 70)

print("Selected model:", best_model_name)

print(
    "F1 Score:",
    round(results[best_model_name]["f1"], 4)
)

print("=" * 70)


# ==========================================
# Save model
# ==========================================

joblib.dump(
    {
        "model": best_model,
        "scaler": scaler,
        "features": list(X.columns)
    },
    MODEL_FILE
)


print("\nModel saved to:")

print(MODEL_FILE)

print("\nTraining completed successfully!")

print("=" * 70)