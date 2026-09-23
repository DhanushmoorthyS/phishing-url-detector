import pandas as pd
from ucimlrepo import fetch_ucirepo


print("Downloading PhiUSIIL dataset...")

# Fetch dataset from UCI
dataset = fetch_ucirepo(id=967)

# Get feature data
X = dataset.data.features

# Get target data
y = dataset.data.targets


print("\nDataset downloaded successfully!")

print("\nNumber of rows:", len(X))

print("\nColumns:")
print(X.columns.tolist())


# ==========================================
# Keep only the original URL
# and the original label
# ==========================================

url_data = pd.DataFrame()

url_data["url"] = X["URL"]

url_data["original_label"] = y["label"]


# ==========================================
# Convert labels
#
# UCI:
# 1 = Legitimate
# 0 = Phishing
#
# Our project:
# 0 = Legitimate
# 1 = Phishing
# ==========================================

url_data["label"] = 1 - url_data["original_label"]


# We don't need the original label anymore
url_data = url_data.drop(columns=["original_label"])


# Remove duplicate URLs
url_data = url_data.drop_duplicates(subset=["url"])


# Remove missing URLs
url_data = url_data.dropna(subset=["url"])


# ==========================================
# Save our cleaned dataset
# ==========================================

output_path = "data/phishing_urls.csv"

url_data.to_csv(output_path, index=False)


print("\nFinal dataset:")
print(url_data.head())

print("\nClass distribution:")
print(url_data["label"].value_counts())

print("\nSaved to:")
print(output_path)