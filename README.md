# Phishing URL Detection System

A machine learning-based phishing URL detection system that analyzes URL characteristics and predicts whether a URL is legitimate or phishing.

## Features

- Machine learning-based URL classification
- URL feature extraction
- Phishing URL detection
- Risk level assessment
- Prediction probability
- Domain evaluation
- Dataset auditing
- Model evaluation
- ROC curve analysis
- Confusion matrix generation
- Feature importance analysis
- Web-based interface using Flask

## Project Structure

```text
phishing-url-detector/
│
├── data/
│   ├── phishing_urls.csv
│   ├── features.csv
│   ├── phishing_model.pkl
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curve.png
│
├── src/
│   ├── audit_dataset.py
│   ├── check_alignment.py
│   ├── domain_evaluation.py
│   ├── download_dataset.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── roc_curve.py
│   ├── test_detector.py
│   ├── train_model.py
│   └── url_analyzer.py
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md