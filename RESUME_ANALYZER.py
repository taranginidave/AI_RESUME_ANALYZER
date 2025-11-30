import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import re
import matplotlib.pyplot as plt

# -------------------------------------------------
#                CUSTOM STYLING
# -------------------------------------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"]{
    background-image: url("https://images.unsplash.com/photo-1521791055366-0d553872125f");
    background-size: cover;
    background-position: center;
}
[data-testid="stHeader"]{
    background-color: rgba(0,0,0,0);
}
.block-container{
    background-color: rgba(255,255,255,0.8);
    padding: 25px;
    border-radius: 15px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------------------------------
#                HELPER FUNCTIONS
# -------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    return text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.lower()

def get_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(similarity * 100, 2)

def improvement_tips(resume_text, jd_text):
    tips = []
    jd_keywords = set(re.findall(r'\b[a-zA-Z]{4,}\b', jd_text.lower()))
    resume_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', resume_text.lower()))

    missing = jd_keywords - resume_words
    if missing:
        tips.append(f"Add relevant missing keywords: {', '.join(list(missing)[:10])}")

    if len(resume_text) < 500:
        tips.append("Your resume seems short — consider adding more projects, skills or achievements.")

    if not re.search(r'(project|experience|education|skills)', resume_text, re.IGNORECASE):
        tips.append("Add mandatory resume sections: Projects, Experience, Education, Skills.")

    return tips

# -------------------------------------------------
#                   PAGE HEADER
# -------------------------------------------------
col1, col2 = st.columns([1, 10])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png", width=80)

with col2:
    st.title("📄 AI-Powered Resume Analyzer")
    st.markdown("### Modern • Smart • Job Matching Engine")

st.write("---")

# -------------------------------------------------
#                INPUT SECTION
# -------------------------------------------------
uploaded_resume = st.file_uploader("📤 Upload Resume (PDF only)", type=["pdf"])
job_desc = st.text_area("📝 Paste Job Description Here", height=200)

if uploaded_resume and job_desc:
    st.info("⏳ Processing resume...")

    resume_text = extract_text_from_pdf(uploaded_resume)
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_desc)

    score = get_similarity(resume_clean, jd_clean)
    tips = improvement_tips(resume_clean, jd_clean)

    # -------------------------------------------------
    #                RESULT SECTION
    # -------------------------------------------------
    st.success(f"🎯 **Resume Match Score: {score}%**")
    st.write("The higher the score, the closer your resume matches the job description.")

    # ---------- PIE CHART ----------
    fig, ax = plt.subplots()
    ax.pie([score, 100 - score],
           labels=["Match", "Gap"],
           autopct='%1.1f%%',
           startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # ---------- TIPS ----------
    st.write("## 🔍 Recommended Improvements")
    for t in tips:
        st.markdown(f"✔️ {t}")

    # -------------------------------------------------
    #              TIMELINE SECTION
    # -------------------------------------------------
    st.write("---")
    st.write("## 🕒 Resume Quality Timeline (Suggested)")

    timeline = """
    🟩 **Step 1:** Add missing job-related keywords  
    🟦 **Step 2:** Improve project details  
    🟪 **Step 3:** Enhance experience descriptions  
    🟧 **Step 4:** Add measurable achievements  
    🟥 **Step 5:** Final formatting & proofreading  
    """

    st.markdown(timeline)

    # -------------------------------------------------
    #       DEBUGGING SECTION (optional)
    # -------------------------------------------------
    with st.expander("📜 Extracted Resume Text"):
        st.text_area("Resume Content", resume_text, height=250)

else:
    st.warning("⚠️ Please upload a resume and paste a job description.")

