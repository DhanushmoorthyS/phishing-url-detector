import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DATA_FILE = "data/features.csv"
MODEL_FILE = "data/phishing_model.pkl"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("        PHISHING URL MODEL EVALUATION")
print("=" * 60)

print("\nLoading feature dataset...")

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["label"])
y = df["label"]

print("Dataset shape:", df.shape)


# --------------------------------------------------
# CREATE SAME TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

print("\nLoading trained model...")

model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
feature_names = model_data["features"]

print("Model loaded successfully.")


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)

# Find the probability belonging to phishing class (1)
phishing_class_index = list(model.classes_).index(1)

phishing_probability = y_probability[:, phishing_class_index]


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "LEGITIMATE",
            "PHISHING"
        ]
    )
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# --------------------------------------------------
# CONFUSION MATRIX VISUALIZATION
# --------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["LEGITIMATE", "PHISHING"]
)

disp.plot()

plt.title("Phishing URL Detection - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "data/confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nConfusion matrix graph saved to:")
print("data/confusion_matrix.png")

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives  :", tp)


# --------------------------------------------------
# ROC-AUC
# --------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    phishing_probability
)

print("\n" + "=" * 60)
print("ROC-AUC SCORE")
print("=" * 60)

print(f"ROC-AUC: {roc_auc:.4f}")


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

if hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print("\nMost important features:\n")

    print(
        importance_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # FEATURE IMPORTANCE GRAPH
    # --------------------------------------------------

    plt.figure(figsize=(10, 7))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        "data/feature_importance.png",
        dpi=300
    )

    print("\nFeature importance graph saved to:")
    print("data/feature_importance.png")

else:

    print(
        "The selected model does not provide "
        "feature_importances_."
    )


print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)