# ScamShield

<p align="center">
  <h1 align="center">ScamShield</h1>
  <p align="center">
    <strong>AI-Powered SMS Scam Detection and Risk Analysis</strong>
  </p>
  <p align="center">
    Detect suspicious messages, assess risk, identify scam categories,
    and understand why a message was flagged.
  </p>
</p>

<p align="center">
  <a href="https://scamshield.streamlit.app/">
    <strong>Live Demo</strong>
  </a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="https://github.com/hemadevireddy/ScamShield">
    <strong>Source Code</strong>
  </a>
</p>

---

## Overview

**ScamShield** is a machine learning-powered SMS security application that
detects potentially fraudulent, spam, and suspicious text messages.

The application uses **Natural Language Processing (NLP)** and a hybrid
machine learning architecture combining:

- Word-level TF-IDF
- Character-level TF-IDF
- Message-level engineered features
- Linear Support Vector Machine (SVM)
- Hyperparameter tuning
- Threshold optimization
- Probability calibration
- Rule-based risk analysis
- Explainable model predictions

Instead of simply returning `Spam` or `Not Spam`, ScamShield provides a
complete security-oriented analysis including:

- Scam probability
- Risk score
- Risk level
- Scam category
- Suspicious indicators
- Message statistics
- Important model features
- Model confidence

---

# Problem Statement

SMS-based scams are increasingly designed to appear like legitimate
communications from banks, companies, delivery services, employers, and
other organizations.

Common examples include:

- Fake prize and lottery messages
- Financial scams
- Phishing messages
- Fake job offers
- Delivery and parcel scams
- Subscription scams
- Account verification scams
- Suspicious promotional messages

Traditional keyword-based filtering can struggle with variations in
language, spelling, formatting, URLs, numbers, and message structure.

ScamShield addresses this problem by combining **text-based NLP features**
with **message-level behavioral and structural features**.

---

# Solution

ScamShield processes an SMS message through multiple stages.

```mermaid
flowchart TD
    A[SMS Message] --> B[Text Preprocessing]

    B --> C[Word TF-IDF]
    B --> D[Character TF-IDF]
    B --> E[Engineered Message Features]

    C --> F[Hybrid Feature Representation]
    D --> F
    E --> F

    F --> G[Linear SVM]
    G --> H[Threshold Optimization]
    H --> I[Scam / Legitimate Prediction]

    I --> J[Probability Calibration]
    J --> K[Scam Probability]

    I --> L[Risk Analysis]
    L --> M[Risk Score]
    L --> N[Risk Level]
    L --> O[Scam Category]

    I --> P[Explainability]
    P --> Q[Important Features]

    K --> R[Final ScamShield Result]
    M --> R
    N --> R
    O --> R
    Q --> R
