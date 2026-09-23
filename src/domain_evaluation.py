import pandas as pd
import numpy as np

from urllib.parse import urlparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


URL_FILE = "data/phishing_urls.csv"
FEATURE_FILE = "data/features.csv"


def get_domain(url):
    """Extract hostname from a URL."""

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        hostname = urlparse(url).hostname

        if hostname is None:
            return ""

        hostname = hostname.lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except Exception:
        return ""


print("=" * 65)
print("          DOMAIN-SEPARATED MODEL EVALUATION")
print("=" * 65)


# ------------------------------------------------------------
# 1. Load original URL dataset
# ------------------------------------------------------------

print("\nLoading URL dataset...")

url_df = pd.read_csv(URL_FILE)

print("Original URL rows:", len(url_df))


# ------------------------------------------------------------
# 2. Reproduce the exact shuffle used during feature extraction
# ------------------------------------------------------------

print("\nReproducing feature-engineering order...")

url_df = url_df.sample(
    n=len(url_df),
    random_state=42
).reset_index(drop=True)


# ------------------------------------------------------------
# 3. Load feature dataset
# ------------------------------------------------------------

print("\nLoading feature dataset...")

feature_df = pd.read_csv(FEATURE_FILE)

print("Feature rows:", len(feature_df))


# ------------------------------------------------------------
# 4. Verify row counts
# ------------------------------------------------------------

if len(url_df) != len(feature_df):
    raise ValueError(
        "URL dataset and feature dataset have different row counts."
    )


# ------------------------------------------------------------
# 5. Extract domains
# ------------------------------------------------------------

print("\nExtracting domains...")

url_df["domain"] = url_df["url"].apply(get_domain)

empty_domains = (url_df["domain"] == "").sum()

print("Empty domains:", empty_domains)

if empty_domains > 0:
    print("Warning: some URLs have no extracted domain.")


# ------------------------------------------------------------
# 6. Prepare features and labels
# ------------------------------------------------------------

X = feature_df.drop(columns=["label"])

y = feature_df["label"]

groups = url_df["domain"]


# ------------------------------------------------------------
# 7. Stratified Group Split
# ------------------------------------------------------------

print("\nCreating domain-separated split...")

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(
        X,
        y,
        groups
    )
)


X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


train_domains = set(groups.iloc[train_indices])
test_domains = set(groups.iloc[test_indices])

overlapping_domains = train_domains.intersection(test_domains)


# ------------------------------------------------------------
# 8. Print split information
# ------------------------------------------------------------

print("\n" + "-" * 65)
print("DOMAIN SPLIT")
print("-" * 65)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("Training domains:", len(train_domains))
print("Testing domains :", len(test_domains))

print("Overlapping domains:", len(overlapping_domains))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ------------------------------------------------------------
# 9. Safety check
# ------------------------------------------------------------

if len(overlapping_domains) != 0:
    raise ValueError(
        "Domain leakage detected! Some domains appear in both sets."
    )

print("\nDomain leakage check: PASSED")


# ------------------------------------------------------------
# 10. Train Random Forest
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 11. Predictions
# ------------------------------------------------------------

print("Generating predictions...")

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 12. Evaluation metrics
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 65)
print("          DOMAIN-SEPARATED RESULTS")
print("=" * 65)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ------------------------------------------------------------
# 13. Classification report
# ------------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "LEGITIMATE",
            "PHISHING"
        ],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 14. Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 15. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("DOMAIN EVALUATION COMPLETE")
print("=" * 65)

print("\nThe test set contains domains that were")
print("not present in the training set.")

print("\nOverlapping domains:", len(overlapping_domains))

print("=" * 65)