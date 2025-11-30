import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import re

# ---------- Helper functions ----------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

def get_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(similarity * 100, 2)

def improvement_tips(resume_text, jd_text):
    tips = []
    jd_keywords = set(re.findall(r'\\b[a-zA-Z]{4,}\\b', jd_text.lower()))
    resume_words = set(re.findall(r'\\b[a-zA-Z]{4,}\\b', resume_text.lower()))

    missing = jd_keywords - resume_words
    if missing:
        tips.append(f"Add relevant keywords: {', '.join(list(missing)[:10])}")

    if len(resume_text) < 500:
        tips.append("Your resume seems short — consider adding more details or achievements.")

    if not re.search(r'project|experience|education|skills', resume_text, re.IGNORECASE):
        tips.append("Include sections like Projects, Experience, Education, and Skills.")

    if not tips:
        tips.append("Your resume aligns well with the job description!")

    return tips

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")
st.title("📄 AI Resume Analyzer & Job Match")

st.write("Upload your **resume (PDF)** and paste a **job description** to check compatibility.")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description", height=200)

if uploaded_resume and job_desc:
    st.info("Processing resume...")

    resume_text = extract_text_from_pdf(uploaded_resume)
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_desc)

    score = get_similarity(resume_clean, jd_clean)
    tips = improvement_tips(resume_clean, jd_clean)

    st.success(f"✅ Match Score: {score}%")
    st.write("### 🔍 Improvement Suggestions:")
    for t in tips:
        st.markdown(f"- {t}")

    with st.expander("📜 Extracted Resume Text (for debugging)"):
        st.text_area("Resume Content", resume_text, height=200)

else:
    st.warning("Please upload your resume and enter a job description to analyze.")