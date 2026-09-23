import pandas as pd

urls = pd.read_csv("data/phishing_urls.csv")
features = pd.read_csv("data/features.csv")

print("=" * 60)
print("DATASET ALIGNMENT CHECK")
print("=" * 60)

print("\nURL dataset shape:")
print(urls.shape)

print("\nFeature dataset shape:")
print(features.shape)

print("\nURL labels:")
print(urls["label"].head(10).tolist())

print("\nFeature labels:")
print(features["label"].head(10).tolist())

print("\nNumber of labels that match:")

matches = (
    urls["label"].reset_index(drop=True)
    ==
    features["label"].reset_index(drop=True)
)

print(matches.sum(), "/", len(matches))

print("\nAll labels aligned:", matches.all())

print("=" * 60)