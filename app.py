import streamlit as st
import joblib
import pdfplumber
import docx
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

model = joblib.load('resume_classifier.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def extract_text_from_pdf(file):
    text = ''
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return ' '.join([para.text for para in doc.paragraphs])

def predict_role(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    scores = dict(zip(model.classes_, probability.round(3)))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return prediction, ranked

def show_pie_chart(ranked):
    labels = [r.replace('_', ' ') for r, _ in ranked]
    sizes = [s for _, s in ranked]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# page config
st.set_page_config(page_title="Resume Classifier", page_icon="👾", layout="centered")

# header
st.markdown("<h1 style='text-align:center;'> Resume Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Upload your resume and get classified instantly.<br>Currently trained to predict between 4 roles - React Developer, SQL Developer, Workday & Peoplesoft.<br>Model capability will expand with more data</p>", unsafe_allow_html=True)
st.divider()

# input toggle
input_type = st.radio("Choose Input Method", ["Upload File (PDF/DOCX)", "Paste Resume Text"], horizontal=True)

text = None

if input_type == "Upload File (PDF/DOCX)":
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'docx'], label_visibility='collapsed')
    if uploaded_file is not None:
        with st.spinner("Extracting text..."):
            if uploaded_file.name.endswith('.pdf'):
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = extract_text_from_docx(uploaded_file)

else:
    pasted_text = st.text_area("Paste your resume text here", height=200)
    if st.button("Classify"):
        if pasted_text.strip():
            text = pasted_text
        else:
            st.warning("Please paste some text first.")

if text:
    with st.spinner("Analyzing resume..."):
        role, ranked = predict_role(text)

    # predicted role badge
    st.markdown("<h3 style='text-align:center;'>Predicted Role</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='text-align:center; background-color:#4CAF50; color:white;
        padding:20px; border-radius:12px; font-size:28px; font-weight:bold;'>
        {role.replace('_', ' ')}
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # pie chart
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Role Probability Distribution")
        show_pie_chart(ranked)
    with col2:
        st.markdown("#### Other Possible Matches")
        for i, (r, score) in enumerate(ranked[1:], 1):
            st.markdown(f"**{i}. {r.replace('_', ' ')}** — {int(score*100)}%")

    st.divider()

    # resume preview
    with st.expander("View Extracted Resume Text"):
        st.write(text[:1000] + '...')