import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


FEATURE_FILE = "data/features.csv"
MODEL_FILE = "data/phishing_model.pkl"
OUTPUT_FILE = "data/roc_curve.png"


print("=" * 60)
print("              ROC CURVE ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load feature dataset
# ------------------------------------------------------------

print("\nLoading feature dataset...")

df = pd.read_csv(FEATURE_FILE)

X = df.drop(columns=["label"])
y = df["label"]

print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 2. Recreate the same test split
# ------------------------------------------------------------

print("\nCreating test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 3. Load trained model
# ------------------------------------------------------------

print("\nLoading trained Random Forest model...")

model_data = joblib.load(MODEL_FILE)

model = model_data["model"]

print("Model loaded successfully.")


# ------------------------------------------------------------
# 4. Get phishing probabilities
# ------------------------------------------------------------

print("\nGenerating probability scores...")

probabilities = model.predict_proba(X_test)

phishing_class_index = list(model.classes_).index(1)

phishing_probability = probabilities[:, phishing_class_index]


# ------------------------------------------------------------
# 5. Calculate ROC curve and AUC
# ------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    phishing_probability
)

roc_auc = roc_auc_score(
    y_test,
    phishing_probability
)


print("\nROC-AUC:", round(roc_auc, 4))


# ------------------------------------------------------------
# 6. Create ROC curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Phishing URL Detection")

plt.legend()

plt.grid(True)

plt.tight_layout()


# ------------------------------------------------------------
# 7. Save graph
# ------------------------------------------------------------

plt.savefig(
    OUTPUT_FILE,
    dpi=300
)

plt.close()


print("\nROC curve saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("ROC CURVE ANALYSIS COMPLETE")
print("=" * 60)