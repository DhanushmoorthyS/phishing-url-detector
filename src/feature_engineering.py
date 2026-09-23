import pandas as pd
import sys
import os

# Add src folder to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from url_analyzer import extract_features


# ==========================================
# Settings
# ==========================================

INPUT_FILE = "data/phishing_urls.csv"
OUTPUT_FILE = "data/features.csv"

# Start with a small sample for testing
SAMPLE_SIZE = 235370


# ==========================================
# Load dataset
# ==========================================

print("=" * 60)
print("       PHISHING URL FEATURE ENGINEERING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total URLs available:", len(df))


# ==========================================
# Take a sample first
# ==========================================

df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
)

print("URLs selected for processing:", len(df))


# ==========================================
# Extract URL features
# ==========================================

print("\nExtracting security features...")
print("Please wait...")


feature_data = df["url"].apply(extract_features)

features_df = pd.DataFrame(feature_data.tolist())


# ==========================================
# Add labels
# ==========================================

features_df["label"] = df["label"].values


# ==========================================
# Save feature dataset
# ==========================================

features_df.to_csv(OUTPUT_FILE, index=False)


# ==========================================
# Display results
# ==========================================

print("\nFeature extraction completed!")

print("\nFeature columns:")
print(features_df.columns.tolist())

print("\nFirst 5 rows:")
print(features_df.head())

print("\nDataset shape:")
print(features_df.shape)

print("\nClass distribution:")
print(features_df["label"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)