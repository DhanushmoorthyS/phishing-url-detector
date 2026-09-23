from predict import predict_url


TEST_URLS = [
    {
        "category": "Legitimate-style",
        "url": "https://www.google.com"
    },
    {
        "category": "Legitimate-style",
        "url": "https://www.microsoft.com"
    },
    {
        "category": "Legitimate-style",
        "url": "https://www.apple.com"
    },
    {
        "category": "Legitimate-style",
        "url": "https://www.amazon.com"
    },

    {
        "category": "Suspicious keywords",
        "url": "https://secure-login-account-verification.example.com"
    },
    {
        "category": "IP address",
        "url": "http://192.168.1.100/login/verify/account"
    },
    {
        "category": "Multiple hyphens",
        "url": "https://secure-login-account-verify.example.com"
    },
    {
        "category": "Long path",
        "url": "https://example.com/login/account/verify/security/confirmation/password/update"
    },
    {
        "category": "@ symbol",
        "url": "https://example.com@192.168.1.100/login"
    },
    {
        "category": "Many digits",
        "url": "https://example123456789.com/login"
    },
    {
        "category": "Multiple subdomains",
        "url": "https://login.verify.account.security.example.com"
    },
    {
        "category": "Short URL style",
        "url": "https://bit.ly/abc123"
    }
]


print("=" * 90)
print("                 PHISHING URL SECURITY TEST SUITE")
print("=" * 90)


for number, test_case in enumerate(TEST_URLS, start=1):

    category = test_case["category"]
    url = test_case["url"]

    print("\n" + "-" * 90)
    print(f"TEST CASE {number}")
    print("-" * 90)

    print("Category   :", category)
    print("URL        :", url)

    try:

        result = predict_url(url)

        print("Prediction :", result["result"])
        print("Risk       :", result["risk"])
        print(
            "Probability:",
            f"{result['phishing_probability']:.2f}%"
        )

        print("\nSecurity Indicators:")

        for indicator in result["indicators"]:
            print(" -", indicator)

    except Exception as error:

        print("ERROR:", error)


print("\n" + "=" * 90)
print("                 TESTING COMPLETE")
print("=" * 90)