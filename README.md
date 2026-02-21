# Resume Classifier (NLP + Machine Learning)

A web app that classifies resumes into job roles using NLP and machine learning. Users can upload a PDF/DOCX or paste resume text to get instant role predictions with confidence scores.

## Features
- Upload resumes (PDF/DOCX) or paste raw text  
- Automatic text extraction and preprocessing  
- Job role prediction with confidence distribution  
- Clean UI for quick evaluation  
- Trained to classify: React Developer, SQL Developer, Workday, PeopleSoft  
- Model improves with additional training data

## Tech Stack
- Python  
- NLP (TF-IDF)  
- Machine Learning with scikit-learn  
- Streamlit (UI)  
- pdfplumber, python-docx (file parsing)

## Models Evaluated
- Logistic Regression  
- Linear SVM  
- Random Forest  
- Multinomial Naive Bayes (final model)

**Final Model:** Multinomial Naive Bayes  
Chosen for strong performance and fast inference on sparse text features.

## Installation (Local Setup)

```bash
git clone <your-repo-url>
cd resume-classifier
pip install -r requirements.txt
streamlit run app.py
