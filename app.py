import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pickle
import re
from scipy.sparse import hstack, csr_matrix
import pandas as pd

# nltk installation if not already present
import nltk
from nltk.data import find

def ensure_nltk_resource(resource_name, download_name=None):
    """
    Check if an NLTK resource is already installed; download if missing.
    """
    try:
        find(resource_name)
    except LookupError:
        nltk.download(download_name or resource_name.split('/')[-1], quiet=True)

ensure_nltk_resource('tokenizers/punkt', 'punkt')
ensure_nltk_resource('corpora/stopwords', 'stopwords')


# Numeric features extractors
def has_url(text):
    return int(bool(re.search(r'http[s]?://|www\.|bit\.ly|tinyurl', str(text), re.IGNORECASE)))

def count_digits(text):
    return sum(c.isdigit() for c in str(text))

def exclamation_count(text):
    return str(text).count('!')

def uppercase_ratio(text):
    s = str(text)
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    upp = sum(1 for c in letters if c.isupper())
    return upp / len(letters)

def has_phone_number(text):
    return int(bool(re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', str(text))))

def punctuation_count(text):
    return sum(1 for c in str(text) if c in string.punctuation)

def contains_keyword_group_1(text):
    keywords = ['action required', 'important', 'account', 'payment', 'security', 'alert', 'verify', 'confirm']
    text_lower = str(text).lower()
    return int(any(keyword in text_lower for keyword in keywords))

def contains_keyword_group_2(text):
    keywords = ['free', 'win', 'prize', 'claim', 'urgent', 'limited time', 'guarantee']
    text_lower = str(text).lower()
    return int(any(keyword in text_lower for keyword in keywords))

def contains_high_urgency_words(text):
    keywords = ['alert', 'suspension', 'frozen', 'dispute', 'deleted', 're-activate', 'action required']
    text_lower = str(text).lower()
    return int(any(keyword in text_lower for keyword in keywords))

def contains_financial_terms(text):
    keywords = ['credit card', 'payment', 'dispute', 'subscription', 'fee', 'balance']
    text_lower = str(text).lower()
    return int(any(keyword in text_lower for keyword in keywords))

def contains_technical_jargon(text):
    keywords = ['device', 'log in', 'unauthorized', 'data', 'cloud storage']
    text_lower = str(text).lower()
    return int(any(keyword in text_lower for keyword in keywords))

@st.cache_resource
def load_vectorizer(path='vectorizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model(path='model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

tfidf = load_vectorizer('vectorizer.pkl')
model = load_model('model.pkl')

ps = PorterStemmer()

# Text preprocessing
@st.cache_data
def transform_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)

# Streamlit UI
st.set_page_config(page_title='SMS Spam Classifier', layout='wide')

st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #6A11CB 0%, #2575FC 100%);
        padding: 1.2rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.8rem;
        box-shadow: 0 0 12px rgba(0, 0, 0, 0.25);
    ">
        <h1 style="
            font-size: 2.2rem;
            font-weight: 700;
            color: white;
            margin-bottom: 0.3rem;
            letter-spacing: 0.5px;
        ">SMS Spam Classifier</h1>
        <p style="
            font-size: 1.05rem;
            color: rgba(255,255,255,0.9);
            margin-top: 0.2rem;
        ">
            Machine learning powered detector that distinguishes between spam and legitimate (ham) SMS messages.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")

card_style = """
    background-color: rgba(255, 255, 255, 0.04);
    padding: 1.4rem 1.6rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 0 8px rgba(0, 0, 0, 0.08);
    min-height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
"""

with col1:
    st.markdown(
        f"""
        <div style="{card_style}">
            <div>
                <h3 style='margin:0 0 0.8rem 0; color:#91b8ff;'>How It Works</h3>
                <p style='font-size:1rem; line-height:1.6; color:#e0e0e0; margin:0 0 0.8rem 0;'>
                    The spam classifier analyzes messages using both <b>semantic</b> and <b>structural</b> features.
                    Text content is processed using TF-IDF vectorization, while numeric indicators such as URL presence,
                    digit count, and urgency keywords capture spam-like patterns.
                </p>
                <ul style='margin:0 0 0 1.2rem; color:#cccccc; padding-left:0.4rem;'>
                    <li>Preprocessing: tokenization, stopword removal, stemming</li>
                    <li>Feature extraction: TF-IDF + numeric cues</li>
                    <li>Classification: trained ML model (Spam / Ham)</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="{card_style}">
            <div>
                <h3 style='margin:0 0 0.8rem 0; color:#91b8ff;'>Model Overview</h3>
                <p style='font-size:1rem; line-height:1.6; color:#e0e0e0; margin:0 0 0.8rem 0;'>
                    The system combines a pre-trained <b>TF-IDF vectorizer</b> and a <b>machine learning classifier</b>.
                    The model evaluates multiple textual and numerical signals to estimate the probability of spam.
                </p>
                <ul style='margin:0 0 0 1.2rem; color:#cccccc; padding-left:0.4rem;'>
                    <li>Vectorizer: TF-IDF (unigram–bigram)</li>
                    <li>Classifier: Pickled model (`model.pkl`)</li>
                    <li>Update: Replace files to retrain or fine-tune</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

with st.sidebar:
    st.title("App Documentation")
    st.caption("SMS Spam Classifier — professional reference")

    # Overview
    with st.expander("Overview"):
        st.write(
            "This application classifies SMS messages as **Spam** or **Ham (not spam)**. "
            "It combines TF-IDF text features with engineered numeric signals (URLs, digits, urgency keywords, etc.) "
            "and applies a pickled classifier to produce a prediction and (when available) probabilities."
        )

    # How it works
    with st.expander("How it works"):
        st.markdown(
            "- **Preprocessing:** Tokenization, stopword removal, and stemming.\n"
            "- **Text features:** TF-IDF vectorization of the preprocessed text.\n"
            "- **Numeric signals:** Hand-crafted features (URL, digit counts, punctuation, urgency keywords, etc.).\n"
            "- **Classification:** Concatenate TF-IDF + numeric features and pass to the pickled model."
        )

    # Inputs & Outputs
    with st.expander("Inputs & Outputs"):
        st.markdown(
            "**Input:** Free-form SMS text (single message).\n\n"
            "**Output:** Prediction label (Spam / Ham). If the model supports probabilities, the app shows spam probability."
        )

    # Model information
    with st.expander("Model & Version"):
        st.markdown(
            "- **Vectorizer:** TF-IDF (loaded from `vectorizer.pkl`).\n"
            "- **Classifier:** Pickled model (loaded from `model.pkl`).\n"
            "- **Notes:** Replace these files to update the model behavior."
        )
        st.caption("If your model supports `predict_proba`, probabilities will be displayed.")

    st.divider()

    # Samples
    st.header("Samples")
    sample_choice = st.selectbox(
        "Choose a sample message to load into the input box",
        [
            "Select...",
            "ATM WITHDRAWAL ALERT: A transaction of $400.00 was attempted in Chicago, IL. If this wasn't you, reply NO now to cancel.",
            "I'm running about 15 minutes late. So sorry! I'll text you when I pull up.",
            "Free [Product Name] Trial! Just pay $1.99 for shipping. We'll cancel your subscription anytime. Reply STOP to opt out.",
            "Just confirming that the payment of $45.50 for the utilities bill was successfully processed today.",
            "Your Netflix subscription has EXPIRED. Click the link to update your billing information or your account will be TERMINATED by tomorrow. http://nf-billing.co/update",
            "Hey, are we still on for tonight?",
            "Reminder: Your electricity bill of $45 is due tomorrow.",
            "SECURITY ALERT: Unauthorized login detected on your device from (45.201.32.11). Verify identity now at: 1-888-220-3300 or risk account SUSPENSION.",
            "Appointment Reminder: You have an appointment with Dr. Smith on 11/15 at 3:00 PM. Please arrive 10 minutes early."
        ]
    )

    st.divider()
    st.header("Options")
    show_feature_breakdown = st.checkbox("Show numeric feature breakdown", value=True)
    show_raw_transformed = st.checkbox("Show transformed text (tokens + stem)", value=False)

    st.divider()

        # Insights & Usage Guide
    st.header("Insights & Usage Guide")

    with st.expander("Interpreting Results"):
        st.markdown(
            "**Spam Prediction (Red)** → Message contains promotional, suspicious, or urgent cues like URLs, financial terms, or excessive punctuation.\n\n"
            "**Ham Prediction (Green)** → Message has natural conversational structure without spam indicators.\n\n"
            "The model uses both textual semantics (TF-IDF) and numeric signals to make this decision."
        )

    with st.expander("Feature Signals Used by Model"):
        st.markdown(
            "- **Structural:** URL presence, digit count, punctuation, capitalization ratio.\n"
            "- **Semantic:** Urgency words (e.g., 'action required', 'alert'), promotional terms ('free', 'win', 'offer').\n"
            "- **Contextual:** Financial, security, or technical references that appear often in scam messages."
        )

    with st.expander("Model Behavior Tips"):
        st.markdown(
            "1. **Short messages** with common words are usually Ham.\n"
            "2. **Long, formatted texts** containing numbers, URLs, or all-caps warnings tend to trigger Spam classification.\n"
            "3. Even small keyword groups (e.g., 'verify account') can strongly influence predictions.\n"
            "4. Accuracy depends on how closely new messages resemble those in the training data."
        )

    with st.expander("Version & Maintenance"):
        st.markdown(
            "- **TF-IDF Vectorizer:** Loaded from `vectorizer.pkl`\n"
            "- **Model:** Loaded from `model.pkl`\n"
            "- To update, replace these files with retrained versions.\n"
            "- Ensure compatibility (same preprocessing and feature order)."
        )

    st.markdown("---")
    st.caption("This guide explains the model’s logic, signals, and reliability considerations for better interpretation.")

with st.form(key='predict_form'):
    sms_input = st.text_area('Enter message here', value=sample_choice if sample_choice != "Select..." else "", height=140)
    submitted = st.form_submit_button('Predict')

if submitted:
    if not sms_input.strip():
        st.warning("Please enter a non-empty message to classify.")
    else:
        # Preprocess, vectorize, and compute numeric features
        transformed_sms = transform_text(sms_input)
        vector_input_text = tfidf.transform([transformed_sms])

        num_features = [
            [
                has_url(sms_input),
                count_digits(sms_input),
                exclamation_count(sms_input),
                uppercase_ratio(sms_input),
                len(nltk.word_tokenize(sms_input)),
                has_phone_number(sms_input),
                punctuation_count(sms_input),
                contains_keyword_group_1(sms_input),
                contains_keyword_group_2(sms_input),
                contains_high_urgency_words(sms_input),
                contains_financial_terms(sms_input),
                contains_technical_jargon(sms_input)
            ]
        ]
        vector_input_num = csr_matrix(num_features)
        vector_input = hstack([vector_input_text, vector_input_num])

        with st.spinner("Running prediction..."):
            try:
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vector_input)[0]
                pred = model.predict(vector_input)[0]
            except Exception as e:
                st.error(f"Prediction error: {e}")
                pred = None
                proba = None

        result_col1, result_col2 = st.columns([2, 1])

        if pred is not None:
            if pred == 1:
                result_col1.error("Prediction: SPAM")
            else:
                result_col1.success("Prediction: HAM (Not Spam)")

            # Show probability if available
            if proba is not None:
                prob_spam = float(proba[1])
                prob_ham = float(proba[0])
                result_col2.metric("Spam probability", f"{prob_spam:.2%}")
                st.progress(min(max(prob_spam, 0.0), 1.0))
            else:
                result_col2.info("Probability not available for this model.")

            if show_raw_transformed:
                st.subheader("Transformed text")
                st.code(transformed_sms)

            if show_feature_breakdown:
                feature_names = [
                    "has_url",
                    "digit_count",
                    "exclamation_count",
                    "uppercase_ratio",
                    "token_count",
                    "has_phone_number",
                    "punctuation_count",
                    "keyword_group_1",
                    "keyword_group_2",
                    "high_urgency_words",
                    "financial_terms",
                    "technical_jargon"
                ]
                feature_values = num_features[0]
                df_features = pd.DataFrame({
                    "feature": feature_names,
                    "value": feature_values
                })
                with st.expander("Numeric feature breakdown"):
                    st.table(df_features)

            # Enhanced Explanation section with visual emphasis
            st.markdown(
                """
                <div style='margin-top:1.5rem;'>
                    <h3 style='color:#91b8ff; margin-bottom:0.8rem;'>Explanation</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            if pred == 1:
                st.markdown(
                    """
                    <div style="
                        background-color: rgba(255, 65, 65, 0.15);
                        border-left: 5px solid #ff4b4b;
                        padding: 1rem 1.2rem;
                        border-radius: 8px;
                        margin-bottom: 1rem;
                    ">
                        <p style="color:#ffcccc; font-size:1rem; line-height:1.6; margin:0;">
                            <strong>Spam Detected:</strong> The classifier identified several spam-like patterns —
                            such as the presence of <b>URLs</b>, <b>promotional or urgency keywords</b>, 
                            excessive <b>digits or punctuation</b>, or abnormal <b>text structure</b>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    """
                    <div style="
                        background-color: rgba(50, 205, 50, 0.12);
                        border-left: 5px solid #4CAF50;
                        padding: 1rem 1.2rem;
                        border-radius: 8px;
                        margin-bottom: 1rem;
                    ">
                        <p style="color:#c9f7c4; font-size:1rem; line-height:1.6; margin:0;">
                            <strong>Ham (Not Spam):</strong> The message appears natural and conversational.
                            It lacks strong spam indicators like promotional phrases, urgency cues, or structured patterns
                            typically seen in spam messages.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


st.markdown("---")
st.caption("App built with Streamlit. Update `vectorizer.pkl` and `model.pkl` in the same folder to change model behavior.")