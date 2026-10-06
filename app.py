%%writefile /content/ScamShield/app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from scipy.sparse import hstack, csr_matrix


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ScamShield",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# GLOBAL STYLING
# ============================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}


/* ---------- HERO ---------- */

.hero {
    background: linear-gradient(
        135deg,
        #0b1f3a 0%,
        #163b68 100%
    );

    padding: 34px 40px;
    border-radius: 18px;
    margin-bottom: 28px;
    box-shadow: 0 8px 30px rgba(15, 23, 42, 0.14);
}

.hero-title {
    color: #ffffff !important;
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 6px;
}

.hero-subtitle {
    color: #dbeafe !important;
    font-size: 17px;
    margin-bottom: 16px;
}

.hero-description {
    color: #e2e8f0 !important;
    font-size: 14px;
    max-width: 700px;
    line-height: 1.6;
}


/* ---------- SECTION HEADINGS ---------- */

.section-title {
    color: var(--text-color) !important;
    font-size: 23px;
    font-weight: 750;
    margin-top: 10px;
    margin-bottom: 14px;
}


/* ---------- INPUT ---------- */

div[data-testid="stTextArea"] textarea {
    color: var(--text-color) !important;
    background-color: var(--secondary-background-color) !important;
    border: 1px solid rgba(128,128,128,0.30) !important;
    border-radius: 13px !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: var(--text-color) !important;
    opacity: 0.55;
}


/* ---------- BUTTON ---------- */

div.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}


/* ---------- RESULT BANNERS ---------- */

.result-scam {
    background: rgba(220, 38, 38, 0.10);
    border: 1px solid rgba(220, 38, 38, 0.35);
    border-left: 5px solid #dc2626;
    padding: 20px 22px;
    border-radius: 12px;
    color: var(--text-color);
}

.result-safe {
    background: rgba(22, 163, 74, 0.10);
    border: 1px solid rgba(22, 163, 74, 0.35);
    border-left: 5px solid #16a34a;
    padding: 20px 22px;
    border-radius: 12px;
    color: var(--text-color);
}

.result-title {
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 5px;
}

.result-description {
    font-size: 14px;
    opacity: 0.78;
}


/* ---------- METRIC CARDS ---------- */

.metric-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 13px;
    padding: 18px;
    min-height: 105px;
}

.metric-label {
    color: var(--text-color);
    opacity: 0.62;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    color: var(--text-color);
    font-size: 25px;
    font-weight: 800;
    margin-top: 7px;
}


/* ---------- CONTENT CARDS ---------- */

.content-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 14px;
    padding: 21px;
    height: 100%;
}


/* ---------- INDICATORS ---------- */

.indicator {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.16);
    color: var(--text-color);
    padding: 11px 13px;
    border-radius: 9px;
    margin-bottom: 8px;
    font-size: 14px;
}


/* ---------- FEATURE ROW ---------- */

.feature-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(128,128,128,0.07);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 8px;
    padding: 10px 13px;
    margin-bottom: 7px;
    color: var(--text-color);
}

.feature-name {
    font-size: 13px;
    font-weight: 600;
}

.feature-score {
    font-size: 13px;
    font-weight: 700;
}


/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: var(--text-color);
    opacity: 0.5;
    font-size: 12px;
    padding-top: 25px;
}


/* ---------- DARK MODE EXTRA CONTRAST ---------- */

@media (prefers-color-scheme: dark) {

    .section-title {
        color: #f8fafc !important;
    }

    .metric-label {
        color: #cbd5e1 !important;
    }

    .metric-value {
        color: #f8fafc !important;
    }

    .feature-name {
        color: #f1f5f9 !important;
    }

    .feature-score {
        color: #93c5fd !important;
    }

    .indicator {
        color: #e2e8f0 !important;
    }
}


/* ---------- RISK / PROBABILITY ---------- */

.risk-wrap {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 14px;
    padding: 20px 22px;
    height: 100%;
}

.risk-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 10px;
}

.risk-label {
    color: var(--text-color);
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .6px;
    opacity: .65;
}

.risk-number {
    color: var(--text-color);
    font-size: 30px;
    font-weight: 850;
}

.progress-track {
    width: 100%;
    height: 10px;
    background: rgba(128,128,128,0.16);
    border-radius: 999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
}

.risk-scale {
    display: flex;
    justify-content: space-between;
    margin-top: 7px;
    color: var(--text-color);
    opacity: .48;
    font-size: 10px;
}

.gauge {
    width: 150px;
    height: 75px;
    margin: 4px auto 8px;
    position: relative;
    overflow: hidden;
}

.gauge-bg {
    position: absolute;
    inset: 0;
    border-radius: 150px 150px 0 0;
    background: conic-gradient(
        from 270deg,
        #16a34a 0deg 54deg,
        #eab308 54deg 108deg,
        #f97316 108deg 144deg,
        #dc2626 144deg 180deg,
        transparent 180deg
    );
}

.gauge-mask {
    position: absolute;
    left: 13px;
    top: 13px;
    width: 124px;
    height: 62px;
    background: var(--secondary-background-color);
    border-radius: 124px 124px 0 0;
}

.gauge-value {
    position: absolute;
    width: 100%;
    bottom: 4px;
    text-align: center;
    color: var(--text-color);
    font-size: 22px;
    font-weight: 850;
}

.highlight-box {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 14px;
    padding: 18px;
    line-height: 1.8;
    color: var(--text-color);
    font-size: 15px;
    white-space: pre-wrap;
    word-break: break-word;
}

.suspicious {
    background: rgba(220, 38, 38, 0.15);
    border-bottom: 2px solid rgba(220,38,38,.55);
    border-radius: 4px;
    padding: 1px 4px;
    font-weight: 700;
}

.safe-word {
    background: rgba(22,163,74,.10);
    border-radius: 4px;
    padding: 1px 3px;
}

.category-badge {
    display: inline-block;
    padding: 7px 11px;
    border-radius: 999px;
    background: rgba(59,130,246,.10);
    border: 1px solid rgba(59,130,246,.25);
    color: var(--text-color);
    font-size: 13px;
    font-weight: 750;
}

.history-item {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 10px;
    padding: 11px 13px;
    margin-bottom: 7px;
}

.history-message {
    color: var(--text-color);
    font-size: 13px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-meta {
    color: var(--text-color);
    opacity: .58;
    font-size: 11px;
    margin-top: 3px;
}

.history-scam {
    color: #dc2626;
    font-weight: 800;
}

.history-safe {
    color: #16a34a;
    font-weight: 800;
}

.explanation-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 9px;
}

.explanation-name {
    width: 190px;
    color: var(--text-color);
    font-size: 12px;
    font-weight: 650;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.explanation-track {
    flex: 1;
    height: 8px;
    background: rgba(128,128,128,.14);
    border-radius: 999px;
    overflow: hidden;
}

.explanation-fill {
    height: 100%;
    background: #dc2626;
    border-radius: 999px;
}

.explanation-score {
    width: 58px;
    text-align: right;
    color: var(--text-color);
    opacity: .72;
    font-size: 11px;
    font-weight: 750;
}

.action-row {
    display: flex;
    gap: 8px;
}

@media (max-width: 800px) {
    .explanation-name { width: 120px; }
    .hero-title { font-size: 32px; }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_DIR = "/content/ScamShield"


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    metadata = joblib.load(
        f"{MODEL_DIR}/metadata.pkl"
    )

    calibrated_model = joblib.load(
        f"{MODEL_DIR}/calibrated_model.pkl"
    )

    final_svm_model = joblib.load(
        f"{MODEL_DIR}/final_svm_model.pkl"
    )

    word_vectorizer = joblib.load(
        f"{MODEL_DIR}/word_vectorizer.pkl"
    )

    char_vectorizer = joblib.load(
        f"{MODEL_DIR}/char_vectorizer.pkl"
    )

    engineered_scaler = joblib.load(
        f"{MODEL_DIR}/engineered_scaler.pkl"
    )

    return (
        metadata,
        calibrated_model,
        final_svm_model,
        word_vectorizer,
        char_vectorizer,
        engineered_scaler
    )


(
    metadata,
    calibrated_model,
    final_svm_model,
    word_vectorizer,
    char_vectorizer,
    engineered_scaler
) = load_models()


# ============================================================
# FEATURE NAMES
# ============================================================

engineered_feature_names = [
    "message_length",
    "word_count",
    "uppercase_count",
    "digit_count",
    "exclamation_count",
    "question_count",
    "currency_symbol_count",
    "url_count",
    "phone_number_count",
    "special_char_count"
]

all_feature_names = np.concatenate([
    word_vectorizer.get_feature_names_out(),
    char_vectorizer.get_feature_names_out(),
    engineered_feature_names
])


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URLTOKEN ",
        text
    )

    text = re.sub(
        r"\b(?:\+?\d[\d\s-]{7,}\d)\b",
        " PHONETOKEN ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def extract_message_features(text):

    text = str(text)

    return pd.Series({

        "message_length":
            len(text),

        "word_count":
            len(text.split()),

        "uppercase_count":
            sum(1 for c in text if c.isupper()),

        "digit_count":
            sum(1 for c in text if c.isdigit()),

        "exclamation_count":
            text.count("!"),

        "question_count":
            text.count("?"),

        "currency_symbol_count":
            len(re.findall(r"[£$€₹]", text)),

        "url_count":
            len(
                re.findall(
                    r"(https?://|www\.|bit\.ly|tinyurl)",
                    text.lower()
                )
            ),

        "phone_number_count":
            len(
                re.findall(
                    r"\b(?:\+?\d[\d\s-]{8,}\d)\b",
                    text
                )
            ),

        "special_char_count":
            len(
                re.findall(
                    r"[^a-zA-Z0-9\s]",
                    text
                )
            )
    })


# ============================================================
# RISK ANALYSIS
# ============================================================

def analyze_scam_risk(message):

    indicators = []
    risk_points = 0

    text = message.lower()


    url_count = len(
        re.findall(
            r"(https?://\S+|www\.\S+)",
            message,
            re.IGNORECASE
        )
    )

    if url_count > 0:

        indicators.append(
            f"Contains {url_count} URL/link"
        )

        risk_points += 25


    currency_count = len(
        re.findall(
            r"[$₹€£]",
            message
        )
    )

    if currency_count > 0:

        indicators.append(
            "Contains financial/currency references"
        )

        risk_points += 15


    digit_count = len(
        re.findall(
            r"\d",
            message
        )
    )

    if digit_count >= 3:

        indicators.append(
            "Contains multiple numbers"
        )

        risk_points += 10


    urgency_words = [
        "urgent",
        "immediately",
        "hurry",
        "act now",
        "limited time",
        "expires",
        "last chance"
    ]

    if any(
        word in text
        for word in urgency_words
    ):

        indicators.append(
            "Uses urgency or pressure language"
        )

        risk_points += 15


    reward_words = [
        "won",
        "winner",
        "prize",
        "reward",
        "cash",
        "free",
        "congratulations",
        "claim"
    ]

    if any(
        word in text
        for word in reward_words
    ):

        indicators.append(
            "Contains prize/reward language"
        )

        risk_points += 15


    action_words = [
        "click",
        "claim",
        "verify",
        "confirm",
        "send",
        "reply",
        "call"
    ]

    if any(
        word in text
        for word in action_words
    ):

        indicators.append(
            "Contains a call-to-action"
        )

        risk_points += 10


    if message.count("!") >= 2:

        indicators.append(
            "Uses repeated exclamation marks"
        )

        risk_points += 5


    return {
        "risk_points": min(risk_points, 100),
        "indicators": indicators
    }


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_category(message):

    text = message.lower()

    categories = {

        "Prize / Lottery Scam": [
            "winner",
            "won",
            "prize",
            "lottery",
            "reward",
            "cash prize",
            "congratulations"
        ],

        "Financial Scam": [
            "bank",
            "account",
            "loan",
            "credit",
            "debit",
            "payment",
            "refund",
            "upi",
            "transaction"
        ],

        "Phishing / Credential Scam": [
            "verify",
            "password",
            "otp",
            "login",
            "credential",
            "account suspended",
            "confirm your account"
        ],

        "Job / Recruitment Scam": [
            "job",
            "salary",
            "hiring",
            "vacancy",
            "work from home",
            "recruitment"
        ],

        "Delivery / Parcel Scam": [
            "parcel",
            "delivery",
            "courier",
            "package",
            "shipment"
        ],

        "Subscription / Service Scam": [
            "subscription",
            "renew",
            "membership",
            "expires",
            "service"
        ]
    }


    scores = {
        category: sum(
            keyword in text
            for keyword in keywords
        )
        for category, keywords in categories.items()
    }


    best_category = max(
        scores,
        key=scores.get
    )


    if scores[best_category] == 0:

        return "General Spam / Suspicious"


    return best_category


# ============================================================
# PREDICTION
# ============================================================

def predict_message(message):

    cleaned = clean_text(message)


    word_features = word_vectorizer.transform(
        [cleaned]
    )


    char_features = char_vectorizer.transform(
        [cleaned]
    )


    engineered = extract_message_features(
        message
    ).to_frame().T


    engineered_scaled = engineered_scaler.transform(
        engineered
    )


    hybrid_features = hstack([
        word_features,
        char_features,
        csr_matrix(engineered_scaled)
    ])


    svm_score = final_svm_model.decision_function(
        hybrid_features
    )[0]


    probability = calibrated_model.predict_proba(
        hybrid_features
    )[0][1]


    prediction = (
        "SCAM"
        if svm_score >= metadata["best_threshold"]
        else "LEGITIMATE"
    )


    risk = analyze_scam_risk(
        message
    )


    # --------------------------------------------------------
    # TOP ML CONTRIBUTIONS
    # --------------------------------------------------------

    coefficients = final_svm_model.coef_[0]

    contributions = (
        hybrid_features.toarray()[0]
        * coefficients
    )

    top_indices = np.argsort(
        contributions
    )[::-1][:8]


    top_features = []

    for index in top_indices:

        if contributions[index] <= 0:
            continue

        name = all_feature_names[index]

        top_features.append(
            (
                name,
                float(contributions[index])
            )
        )


    return {

        "prediction": prediction,

        "probability": probability,

        "svm_score": svm_score,

        "risk_score": risk["risk_points"],

        "indicators": risk["indicators"],

        "category": detect_category(message),

        "features": engineered,

        "top_features": top_features
    }



# ============================================================
# UI HELPERS
# ============================================================

def highlight_suspicious_text(message):
    """
    Visually highlights common risk-bearing phrases in the original message.
    This is a presentation layer; the ML prediction remains unchanged.
    """
    escaped = (
        str(message)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    patterns = [
        r"https?://[^\s<]+",
        r"www\.[^\s<]+",
        r"\b(?:bit\.ly|tinyurl)\S*",
        r"[$₹€£]\s?[\d,]+(?:\.\d+)?",
        r"\b(?:urgent|immediately|hurry|act now|limited time|expires|last chance)\b",
        r"\b(?:won|winner|prize|reward|cash|free|congratulations|claim)\b",
        r"\b(?:click|verify|confirm|send|reply|call)\b",
    ]

    for pattern in patterns:
        escaped = re.sub(
            pattern,
            lambda m: f'<span class="suspicious">{m.group(0)}</span>',
            escaped,
            flags=re.IGNORECASE
        )

    return escaped.replace("\n", "<br>")


def risk_level_from_score(prediction, risk_score):
    if prediction == "SCAM" or risk_score >= 80:
        return "HIGH"
    if risk_score >= 40:
        return "ELEVATED"
    if risk_score >= 20:
        return "MEDIUM"
    return "LOW"


def risk_gauge_html(score):
    score = max(0, min(100, int(score)))
    return f"""
    <div class="risk-wrap">
        <div class="risk-top">
            <span class="risk-label">Risk Score</span>
            <span class="risk-number">{score}/100</span>
        </div>
        <div class="gauge">
            <div class="gauge-bg"></div>
            <div class="gauge-mask"></div>
            <div class="gauge-value">{score}</div>
        </div>
        <div class="risk-scale">
            <span>Low</span><span>Medium</span><span>High</span>
        </div>
    </div>
    """


def probability_html(probability, prediction):
    pct = max(0.0, min(100.0, probability * 100))
    fill = "#dc2626" if prediction == "SCAM" else "#16a34a"
    return f"""
    <div class="risk-wrap">
        <div class="risk-top">
            <span class="risk-label">Scam Probability</span>
            <span class="risk-number">{pct:.1f}%</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill"
                 style="width:{pct:.1f}%;background:{fill};"></div>
        </div>
        <div class="risk-scale">
            <span>0%</span><span>50%</span><span>100%</span>
        </div>
        <div style="margin-top:18px;color:var(--text-color);opacity:.62;font-size:12px;">
            Calibrated probability from the trained ML model
        </div>
    </div>
    """


# ============================================================
# SESSION HISTORY
# ============================================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">ScamShield</div>
    <div class="hero-subtitle">Intelligent SMS Scam Detection</div>
    <div class="hero-description">
        Analyze suspicious messages using a hybrid machine-learning model
        combining word patterns, character patterns and message-level signals.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# ANALYZER
# ============================================================

st.markdown(
    '<div class="section-title">Analyze a Message</div>',
    unsafe_allow_html=True
)

message = st.text_area(
    "Message",
    height=155,
    placeholder="Paste the SMS or message you want to analyze...",
    label_visibility="collapsed"
)

a1, a2 = st.columns([5, 1])

with a1:
    analyze_button = st.button(
        "Analyze Message",
        type="primary",
        use_container_width=True
    )

with a2:
    clear_button = st.button(
        "Clear",
        use_container_width=True
    )

if clear_button:
    st.session_state.analysis_history = []
    st.rerun()


# ============================================================
# ANALYSIS RESULT
# ============================================================

if analyze_button:

    if not message.strip():
        st.warning("Please enter a message before analyzing.")

    else:
        result = predict_message(message)

        probability = result["probability"]
        prediction = result["prediction"]
        risk_score = result["risk_score"]
        risk_level = risk_level_from_score(prediction, risk_score)

        # Keep a compact history for the current session.
        st.session_state.analysis_history.insert(
            0,
            {
                "message": message.strip(),
                "prediction": prediction,
                "probability": probability,
                "category": result["category"],
                "risk_score": risk_score,
            }
        )
        st.session_state.analysis_history = st.session_state.analysis_history[:5]

        st.divider()

        # --------------------------------------------------------
        # RESULT BANNER
        # --------------------------------------------------------

        if prediction == "SCAM":
            st.markdown("""
            <div class="result-scam">
                <div class="result-title">Scam Detected</div>
                <div class="result-description">
                    The model detected strong characteristics associated
                    with scam or spam messages.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-safe">
                <div class="result-title">Appears Legitimate</div>
                <div class="result-description">
                    The model did not detect strong scam characteristics
                    in this message.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # --------------------------------------------------------
        # PROBABILITY + RISK GAUGE
        # --------------------------------------------------------

        p1, p2 = st.columns(2)

        with p1:
            st.markdown(
                probability_html(probability, prediction),
                unsafe_allow_html=True
            )

        with p2:
            st.markdown(
                risk_gauge_html(risk_score),
                unsafe_allow_html=True
            )

        st.write("")

        # --------------------------------------------------------
        # CATEGORY / RISK LEVEL
        # --------------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value">{risk_level}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Category</div>
                    <div style="margin-top:10px;">
                        <span class="category-badge">{result["category"]}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">ML Decision Score</div>
                    <div class="metric-value">{result["svm_score"]:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # --------------------------------------------------------
        # RISK INDICATORS + MESSAGE STATISTICS
        # --------------------------------------------------------

        left, right = st.columns(2)

        with left:
            st.markdown(
                '<div class="section-title">Risk Indicators</div>',
                unsafe_allow_html=True
            )

            if result["indicators"]:
                for indicator in result["indicators"]:
                    st.markdown(
                        f'<div class="indicator">{indicator}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.success("No major rule-based risk indicators detected.")

        with right:
            st.markdown(
                '<div class="section-title">Message Statistics</div>',
                unsafe_allow_html=True
            )

            stats = result["features"].iloc[0]

            stats_df = pd.DataFrame({
                "Feature": [
                    "Message length",
                    "Word count",
                    "Digits",
                    "URLs",
                    "Currency symbols",
                    "Exclamation marks",
                    "Question marks"
                ],
                "Value": [
                    int(stats["message_length"]),
                    int(stats["word_count"]),
                    int(stats["digit_count"]),
                    int(stats["url_count"]),
                    int(stats["currency_symbol_count"]),
                    int(stats["exclamation_count"]),
                    int(stats["question_count"])
                ]
            })

            st.dataframe(
                stats_df,
                hide_index=True,
                use_container_width=True
            )

        # --------------------------------------------------------
        # SUSPICIOUS TEXT
        # --------------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">Suspicious Content</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Potentially risk-bearing phrases are highlighted based on "
            "the application's rule-based indicators."
        )

        st.markdown(
            f'<div class="highlight-box">{highlight_suspicious_text(message)}</div>',
            unsafe_allow_html=True
        )

        # --------------------------------------------------------
        # WHY WAS IT FLAGGED?
        # --------------------------------------------------------

        if result["top_features"]:
            st.divider()

            st.markdown(
                '<div class="section-title">Why Was It Flagged?</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Strongest positive model features contributing to the "
                "scam classification."
            )

            max_score = max(score for _, score in result["top_features"])

            for feature, score in result["top_features"]:
                width = max(8, min(100, (score / max_score) * 100))

                st.markdown(
                    f"""
                    <div class="explanation-bar">
                        <div class="explanation-name">{feature}</div>
                        <div class="explanation-track">
                            <div class="explanation-fill"
                                 style="width:{width:.1f}%;"></div>
                        </div>
                        <div class="explanation-score">+{score:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # --------------------------------------------------------
        # MODEL CONFIDENCE
        # --------------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">Model Confidence</div>',
            unsafe_allow_html=True
        )

        mc1, mc2 = st.columns(2)

        with mc1:
            st.metric(
                "Calibrated Scam Probability",
                f"{probability * 100:.2f}%"
            )

        with mc2:
            st.metric(
                "ML Decision Score",
                f"{result['svm_score']:.4f}"
            )


# ============================================================
# ANALYSIS HISTORY
# ============================================================

if st.session_state.analysis_history:
    st.divider()

    st.markdown(
        '<div class="section-title">Recent Analyses</div>',
        unsafe_allow_html=True
    )

    for item in st.session_state.analysis_history:
        status_class = (
            "history-scam"
            if item["prediction"] == "SCAM"
            else "history-safe"
        )

        st.markdown(
            f"""
            <div class="history-item">
                <div class="history-message">{item["message"]}</div>
                <div class="history-meta">
                    <span class="{status_class}">{item["prediction"]}</span>
                    &nbsp; · &nbsp;
                    {item["probability"] * 100:.1f}% probability
                    &nbsp; · &nbsp;
                    {item["category"]}
                    &nbsp; · &nbsp;
                    Risk {item["risk_score"]}/100
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    ScamShield · Hybrid TF-IDF + Engineered Features + Linear SVM
    <br>
    Accuracy 99.23% · F1 96.83% · ROC-AUC 99.76%
</div>
""", unsafe_allow_html=True)
