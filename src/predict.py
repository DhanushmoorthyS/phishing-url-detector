import joblib
import pandas as pd

from url_analyzer import extract_features


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_FILE = "data/phishing_model.pkl"

model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
feature_names = model_data["features"]


# --------------------------------------------------
# RISK LEVEL
# --------------------------------------------------

def get_risk_level(probability):

    if probability < 0.30:
        return "LOW"

    elif probability < 0.70:
        return "MEDIUM"

    else:
        return "HIGH"


# --------------------------------------------------
# SECURITY INDICATORS
# --------------------------------------------------

def analyze_indicators(features):

    indicators = []

    if features["has_ip"] == 1:
        indicators.append(
            "URL uses an IP address instead of a domain name"
        )

    if features["has_at_symbol"] == 1:
        indicators.append(
            "URL contains an @ symbol"
        )

    if features["has_https"] == 0:
        indicators.append(
            "URL does not use HTTPS"
        )

    if features["suspicious_keyword_count"] > 0:
        indicators.append(
            f"Contains {features['suspicious_keyword_count']} suspicious keyword(s)"
        )

    if features["is_shortened_url"] == 1:
        indicators.append(
            "URL uses a known URL shortening service"
        )

    if features["subdomain_count"] >= 3:
        indicators.append(
            "URL contains multiple subdomains"
        )

    if features["url_length"] > 75:
        indicators.append(
            "URL is unusually long"
        )

    if features["special_character_count"] >= 5:
        indicators.append(
            "URL contains many special characters"
        )

    if features["hyphen_count"] >= 3:
        indicators.append(
            "URL contains multiple hyphens"
        )

    if features["digit_count"] >= 5:
        indicators.append(
            "URL contains many digits"
        )

    if features["path_length"] >= 30:
        indicators.append(
            "URL contains a long path"
        )
        
    if not indicators:
        indicators.append(
            "No obvious lexical indicators detected"
        )

    return indicators


# --------------------------------------------------
# FEATURE DISPLAY NAMES
# --------------------------------------------------

FEATURE_DISPLAY_NAMES = {

    "url_length": "URL Length",

    "domain_length": "Domain Length",

    "path_length": "Path Length",

    "dot_count": "Dot Count",

    "hyphen_count": "Hyphen Count",

    "digit_count": "Digit Count",

    "special_character_count": "Special Character Count",

    "has_https": "Uses HTTPS",

    "has_ip": "Uses IP Address",

    "has_at_symbol": "Contains @ Symbol",

    "query_parameter_count": "Query Parameters",

    "has_fragment": "Has Fragment",

    "subdomain_count": "Subdomain Count",

    "suspicious_keyword_count": "Suspicious Keyword Count",

    "is_shortened_url": "Shortened URL"
}


# --------------------------------------------------
# FORMAT FEATURES FOR WEB DISPLAY
# --------------------------------------------------

def format_features(features):

    formatted_features = []

    for feature_name, value in features.items():

        display_name = FEATURE_DISPLAY_NAMES.get(
            feature_name,
            feature_name
        )

        # Convert binary features into Yes / No
        if feature_name in [
            "has_https",
            "has_ip",
            "has_at_symbol",
            "has_fragment",
            "is_shortened_url"
        ]:

            display_value = "Yes" if value == 1 else "No"

        else:

            display_value = value

        formatted_features.append({
            "name": display_name,
            "value": display_value
        })

    return formatted_features


# --------------------------------------------------
# PREDICT URL
# --------------------------------------------------

def predict_url(url):

    # Extract URL features
    features = extract_features(url)

    # Convert features to DataFrame
    feature_df = pd.DataFrame([features])

    # Keep exactly the features used during training
    feature_df = feature_df[feature_names]

    # Make prediction
    prediction = model.predict(feature_df)[0]

    # Get class probabilities
    probabilities = model.predict_proba(feature_df)[0]

    # Find probability belonging to phishing class (1)
    phishing_class_index = list(model.classes_).index(1)

    phishing_probability = probabilities[
        phishing_class_index
    ]

    # Convert prediction to readable result
    result = (
        "PHISHING"
        if prediction == 1
        else "LEGITIMATE"
    )

    probability_percent = phishing_probability * 100

    # Determine risk
    risk = get_risk_level(
        phishing_probability
    )

    # Analyze security indicators
    indicators = analyze_indicators(
        features
    )

    # Format features for frontend
    formatted_features = format_features(
        features
    )

    return {

        "url": url,

        "prediction": int(prediction),

        "result": result,

        "risk": risk,

        "phishing_probability": round(
            probability_percent,
            2
        ),

        "features": features,

        "formatted_features": formatted_features,

        "indicators": indicators
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_urls = [

        "https://www.google.com",

        "http://192.168.1.100/login/verify/account",

        "https://secure-login-account-verification.example.com"
    ]

    for url in test_urls:

        print("\n" + "=" * 65)

        result = predict_url(url)

        print("URL:")
        print(result["url"])

        print("\nPrediction:")
        print(result["result"])

        print("Risk Level:")
        print(result["risk"])

        print(
            "Phishing Probability:",
            result["phishing_probability"],
            "%"
        )

        print("\nSecurity Indicators:")

        for indicator in result["indicators"]:
            print(" -", indicator)

        print("\nExtracted Features:")

        for feature in result["formatted_features"]:
            print(
                f" - {feature['name']}: "
                f"{feature['value']}"
            )