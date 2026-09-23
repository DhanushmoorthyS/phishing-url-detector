import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/phishing_urls.csv"


def get_domain(url):
    """Extract a normalized hostname from a URL."""
    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        hostname = urlparse(url).hostname

        if hostname is None:
            return ""

        hostname = hostname.lower()

        # Treat www.example.com and example.com as the same domain
        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except Exception:
        return ""


print("=" * 65)
print("              DATASET SPLIT AUDIT")
print("=" * 65)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total URLs:", len(df))


# ---------------------------------------------------------
# Reproduce the same sampling used during feature engineering
# ---------------------------------------------------------

df = df.sample(
    n=len(df),
    random_state=42
).reset_index(drop=True)


# ---------------------------------------------------------
# Reproduce the same train/test split used during training
# ---------------------------------------------------------

indices = range(len(df))

train_indices, test_indices = train_test_split(
    list(indices),
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)


train_df = df.iloc[train_indices].copy()
test_df = df.iloc[test_indices].copy()


# ---------------------------------------------------------
# Extract normalized domains
# ---------------------------------------------------------

train_df["domain"] = train_df["url"].apply(get_domain)
test_df["domain"] = test_df["url"].apply(get_domain)


train_domains = set(train_df["domain"])
test_domains = set(test_df["domain"])


# Remove empty domains
train_domains.discard("")
test_domains.discard("")


overlapping_domains = train_domains.intersection(test_domains)


# ---------------------------------------------------------
# Exact URL overlap
# ---------------------------------------------------------

train_urls = set(train_df["url"])
test_urls = set(test_df["url"])

overlapping_urls = train_urls.intersection(test_urls)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "-" * 65)
print("TRAIN / TEST SPLIT")
print("-" * 65)

print("Training samples:", len(train_df))
print("Testing samples :", len(test_df))


print("\n" + "-" * 65)
print("DOMAIN ANALYSIS")
print("-" * 65)

print("Unique training domains:", len(train_domains))
print("Unique testing domains :", len(test_domains))
print("Domains appearing in BOTH:", len(overlapping_domains))


print("\n" + "-" * 65)
print("EXACT URL ANALYSIS")
print("-" * 65)

print("Exact URLs appearing in BOTH:", len(overlapping_urls))


# ---------------------------------------------------------
# Show examples of overlapping domains
# ---------------------------------------------------------

if overlapping_domains:

    print("\nExamples of overlapping domains:")

    for domain in sorted(overlapping_domains)[:20]:
        print(" -", domain)

else:

    print("\nNo overlapping domains found.")


print("\n" + "=" * 65)
print("AUDIT COMPLETE")
print("=" * 65)