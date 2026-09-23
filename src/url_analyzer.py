from urllib.parse import urlparse, parse_qs
import ipaddress


# ==========================================
# Suspicious keywords
# ==========================================

SUSPICIOUS_KEYWORDS = [
    "login",
    "log-in",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "account",
    "password",
    "secure",
    "security",
    "bank",
    "update",
    "confirm",
    "credential",
    "authenticate"
]


# ==========================================
# Possible URL shortening services
# ==========================================

SHORTENING_SERVICES = [
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly"
]


# ==========================================
# Check whether hostname is an IP address
# ==========================================

def is_ip_address(hostname):

    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1

    except ValueError:
        return 0


# ==========================================
# Extract features from URL
# ==========================================

def extract_features(url):

    parsed_url = urlparse(url)

    hostname = parsed_url.hostname or ""
    path = parsed_url.path or ""
    query = parsed_url.query or ""

    # --------------------------------------
    # Basic URL features
    # --------------------------------------

    url_length = len(url)

    domain_length = len(hostname)

    path_length = len(path)

    dot_count = url.count(".")

    hyphen_count = url.count("-")

    digit_count = sum(character.isdigit() for character in url)

    special_character_count = sum(
        1 for character in url
        if character in "@?=&_%;$"
    )

    # --------------------------------------
    # HTTPS
    # --------------------------------------

    has_https = 1 if parsed_url.scheme.lower() == "https" else 0

    # --------------------------------------
    # IP address
    # --------------------------------------

    has_ip = is_ip_address(hostname)

    # --------------------------------------
    # @ symbol
    # --------------------------------------

    has_at_symbol = 1 if "@" in url else 0

    # --------------------------------------
    # Query parameters
    # --------------------------------------

    query_parameter_count = len(parse_qs(query))

    # --------------------------------------
    # Fragment
    # --------------------------------------

    has_fragment = 1 if parsed_url.fragment else 0

    # --------------------------------------
    # Subdomain count
    # --------------------------------------

    hostname_parts = hostname.split(".")

    if has_ip:
        subdomain_count = 0
    else:
        subdomain_count = max(len(hostname_parts) - 2, 0)

    # --------------------------------------
    # Suspicious keywords
    # --------------------------------------

    found_keywords = [
        keyword
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in url.lower()
    ]

    suspicious_keyword_count = len(found_keywords)

    # --------------------------------------
    # URL shortening service
    # --------------------------------------

    is_shortened_url = 0

    for service in SHORTENING_SERVICES:

        if service in hostname.lower():
            is_shortened_url = 1
            break

    # --------------------------------------
    # Return numerical features
    # --------------------------------------

    return {
        "url_length": url_length,
        "domain_length": domain_length,
        "path_length": path_length,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "digit_count": digit_count,
        "special_character_count": special_character_count,
        "has_https": has_https,
        "has_ip": has_ip,
        "has_at_symbol": has_at_symbol,
        "query_parameter_count": query_parameter_count,
        "has_fragment": has_fragment,
        "subdomain_count": subdomain_count,
        "suspicious_keyword_count": suspicious_keyword_count,
        "is_shortened_url": is_shortened_url
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    test_url = "https://www.google.com"

    features = extract_features(test_url)

    print("=" * 60)
    print("          PHISHING URL FEATURE ANALYZER")
    print("=" * 60)

    print("\nURL:")
    print(test_url)

    print("\nExtracted Features:")

    for feature, value in features.items():
        print(f"{feature:30} : {value}")

    print("=" * 60)