from flask import Flask, render_template, request
import sys
import os

# Add the src folder to Python's import path
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)

from predict import predict_url

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    url = ""

    if request.method == "POST":
        url = request.form.get("url", "").strip()

        if not url:
            error = "Please enter a URL."
        else:
            try:
                result = predict_url(url)
            except Exception as e:
                error = f"Unable to analyze URL: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        url=url
    )


if __name__ == "__main__":
    print("=" * 60)
    print("       PHISHING URL DETECTOR")
    print("=" * 60)

    print("\nStarting Flask server...")
    print("\nOpen your browser and go to:")
    print("http://127.0.0.1:5000")

    print("\nPress CTRL+C to stop the server.")

    app.run(debug=True)