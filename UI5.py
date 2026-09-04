import streamlit as st
import json
import re
import asyncio
from typing import List
import ollama
from pydantic import BaseModel, Field

# ----------------------------------------------------
# 0. Hardcoded Target Job Description
# ----------------------------------------------------
TARGET_JOB_DESCRIPTION = """
Job Title: Financial Data Analyst (Quantitative Analytics)
Department: Quantitative Research & Financial Analytics
Employment Type: Full-Time

Role Overview:
We are looking for a rigorous, mathematically minded Financial Data Analyst to bridge the gap between complex financial datasets and strategic decision-making. In this role, you will design statistical models, analyze market/transactional data, and build reproducible quantitative frameworks to evaluate risk, forecast performance, and optimize portfolio returns.

Key Responsibilities:
- Quantitative Modeling & Forecasting: Develop, backtest, and refine predictive statistical models, time-series forecasts (ARIMA, GARCH), and risk metrics (VaR, stress-testing, Sharpe ratio).
- Financial Data Engineering: Ingest, clean, and normalize structured and unstructured financial data (tick data, balance sheets, macroeconomic indicators, order books) across various databases and API feeds.
- SQL & Data Extraction: Write high-performance SQL queries involving window functions, indexing, and CTEs to extract insights from relational databases and cloud warehouses.
- Advanced Financial Modeling: Build robust financial models and automated reporting systems in Excel utilizing Power Query, VBA/macros, and dynamic matrix formulas.
- Algorithmic Scripting: Implement data processing pipelines and exploratory quantitative analyses in Python (Pandas, NumPy, SciPy, statsmodels) or R.
- Performance & Risk Dashboards: Design and maintain automated risk and performance attribution dashboards in Power BI, Tableau, or Dash/Streamlit for trading and executive teams.
- Model Validation & Integrity: Ensure data hygiene, investigate pricing or accounting anomalies, and validate underlying statistical assumptions to minimize model risk.

Required Qualifications:
- Education: Bachelor’s or Master’s degree in Finance, Economics, Quantitative Finance, Applied Mathematics, Statistics, Computer Science, or a related discipline.
- Core Technical Stack: Python or R (pandas, numpy, statsmodels, scipy), Advanced SQL, Advanced Excel (financial modeling, scenario analysis, pivot tables).
- Financial Acumen: Strong understanding of corporate finance, valuation methods (DCF, multiples), capital markets, derivatives, and fixed-income/equity instruments.
- Statistical Foundations: Solid grounding in probability, linear regression, multivariate analysis, hypothesis testing, and time-series analysis.

Preferred Qualifications:
- Progress toward or completion of relevant professional credentials (e.g., CFA, FRM).
- Experience with market data terminals and APIs (e.g., Bloomberg B-PIPE/API, FactSet, Refinitiv, Quandl/Nasdaq Data Link).
- Exposure to financial machine learning concepts (e.g., classification, random forests, clustering) or factor modeling.
- Working knowledge of cloud data warehouses (Snowflake, BigQuery) and version control via Git/GitHub.
"""

# ----------------------------------------------------
# 1. Non-Nullable Schema
# ----------------------------------------------------
class ResumeData(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Candidate's email address")
    phone: str = Field(description="Candidate's phone number")
    years_of_experience: float = Field(description="Total calculated work experience in years")
    skills: List[str] = Field(description="All technical skills, programming languages, and analytical tools")
    education: List[str] = Field(description="List of degrees, universities, or schools attended")
    last_3_job_titles: List[str] = Field(description="List of job titles held by candidate")

# ----------------------------------------------------
# 2. Regex Fallback Helpers
# ----------------------------------------------------
def extract_email_fallback(text: str) -> str:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else ""

def extract_phone_fallback(text: str) -> str:
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,5}", text)
    return match.group(0).strip() if match else ""

# ----------------------------------------------------
# 3. PDF Text Extraction (Bytes for Streamlit)
# ----------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes (used by Streamlit file uploader)."""
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF is required. Install via 'pip install pymupdf'")
    text = ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# ----------------------------------------------------
# 4. Async Extraction Function
# ----------------------------------------------------
async def parse_resume_local_async(raw_text: str, model_name: str = "qwen2.5-coder:7b") -> ResumeData:
    system_prompt = (
        "You are an expert recruitment parser. Extract all candidate information from the resume text into the required JSON format.\n\n"
        "EXTRACTION RULES:\n"
        "- NAME: Found at the very beginning of the document.\n"
        "- EMAIL & PHONE: Extract exactly as printed near the contact header.\n"
        "- EDUCATION: Extract degree name and institution as a list of strings.\n"
        "- SKILLS: Extract all technical, coding, software, and financial tools.\n"
        "- YEARS OF EXPERIENCE: Calculate the total career duration across all positions.\n"
        "- LAST 3 JOB TITLES: Extract the recent job designations."
    )

    response = await asyncio.to_thread(
        ollama.chat,
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract all details from this resume text:\n\n{raw_text}"},
        ],
        format=ResumeData.model_json_schema(),
        options={"temperature": 0.0, "num_ctx": 4096},
    )

    result = ResumeData.model_validate_json(response.message.content)

    if not result.email:
        result.email = extract_email_fallback(raw_text)
    if not result.phone:
        result.phone = extract_phone_fallback(raw_text)

    result.last_3_job_titles = result.last_3_job_titles[:3]
    return result

# ----------------------------------------------------
# 5. Match Evaluation Schema
# ----------------------------------------------------
class MatchEvaluation(BaseModel):
    match_score: int = Field(description="Match score between 0 and 100")
    reasons_for_match: List[str] = Field(description="List of reasons why the resume matches")
    reasons_for_mismatch: List[str] = Field(description="List of reasons why the resume does not fully match")
    missing_skills: List[str] = Field(description="List of specific skills missing from the resume")

# ----------------------------------------------------
# 6. Helper to Identify Empty Details
# ----------------------------------------------------
def get_empty_or_missing_details(data: ResumeData) -> List[str]:
    missing = []
    if not data.name or data.name.strip() == "": missing.append("Candidate name is missing")
    if not data.email: missing.append("Email address is missing")
    if not data.phone: missing.append("Phone number is missing")
    if data.years_of_experience <= 0: missing.append("Years of experience is missing or zero")
    if not data.skills: missing.append("No technical skills listed")
    if not data.education: missing.append("Education details are missing")
    if not data.last_3_job_titles: missing.append("Job titles are missing")
    return missing

# ----------------------------------------------------
# 7. Async Matcher Function
# ----------------------------------------------------
async def match_resume_to_job_async(resume_data: ResumeData, empty_details: List[str], job_description: str, model_name: str = "qwen2.5-coder:7b") -> MatchEvaluation:
    system_prompt = (
        "You are an expert recruitment matcher. Compare the candidate's parsed resume data with the provided job description.\n\n"
        "EVALUATION RULES:\n"
        "- MATCH SCORE: Provide an integer score from 0 to 100. Deduct points for missing critical details and missing skills.\n"
        "- REASONS FOR MATCH: Highlight specific skills, experiences, or qualifications that align.\n"
        "- REASONS FOR MISMATCH: Highlight areas where the candidate falls short.\n"
        "- MISSING SKILLS: List specific technical skills mentioned in the JD but absent from the resume."
    )

    resume_json = resume_data.model_dump_json(indent=2)
    empty_details_str = ", ".join(empty_details) if empty_details else "None"

    response = await asyncio.to_thread(
        ollama.chat,
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Resume Data:\n{resume_json}\n\nEmpty/Missing Resume Details:\n{empty_details_str}\n\nJob Description:\n{job_description}"},
        ],
        format=MatchEvaluation.model_json_schema(),
        options={"temperature": 0.0, "num_ctx": 4096},
    )

    return MatchEvaluation.model_validate_json(response.message.content)

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Resume Matcher", layout="wide")

st.title("📄 Resume to Job Description Matcher")
st.markdown("Target Role: **Financial Data Analyst (Quantitative Analytics)**")
st.divider()

# Input Section
col1, col2 = st.columns([1, 1])
with col1:
    input_method = st.radio("How would you like to provide the resume?", ("Upload PDF", "Paste Text"))

raw_text = ""

with col2:
    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file is not None:
            raw_text = extract_text_from_pdf_bytes(uploaded_file.getvalue())
    else:
        raw_text = st.text_area("Paste resume text here:", height=250, placeholder="Paste the candidate's resume text...")

# Process Button
if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
    if not raw_text.strip():
        st.error("Please provide a resume (either upload a PDF or paste text).")
    else:
        # 1. Parse Resume
        with st.spinner("🤖 Extracting and parsing resume data..."):
            data = asyncio.run(parse_resume_local_async(raw_text))
            
        # 2. Identify empty details
        empty_details = get_empty_or_missing_details(data)
        
        # 3. Evaluate Match
        with st.spinner("📊 Evaluating match against Financial Data Analyst role..."):
            match_eval = asyncio.run(match_resume_to_job_async(data, empty_details, TARGET_JOB_DESCRIPTION))
            
        # Display Results
        st.success("✅ Analysis Complete!")
        st.divider()
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Match Score", f"{match_eval.match_score}/100")
        m2.metric("Experience", f"{data.years_of_experience} Years")
        m3.metric("Skills Identified", len(data.skills))
        
        st.subheader(f"👤 Candidate: {data.name}")
        st.write(f"📧 {data.email}  |  📞 {data.phone}")
        st.write(f"🎓 **Education:** {', '.join(data.education) if data.education else 'Not specified'}")
        st.write(f"💼 **Recent Titles:** {', '.join(data.last_3_job_titles) if data.last_3_job_titles else 'Not specified'}")
        
        st.divider()
        
        # Detailed Expanders
        with st.expander("✅ Reasons for Match", expanded=True):
            if match_eval.reasons_for_match:
                for reason in match_eval.reasons_for_match:
                    st.write(f"- {reason}")
            else:
                st.info("No specific matching reasons identified.")
                
        with st.expander("❌ Reasons for Mismatch", expanded=True):
            if match_eval.reasons_for_mismatch:
                for reason in match_eval.reasons_for_mismatch:
                    st.write(f"- {reason}")
            else:
                st.info("No mismatches identified.")
                
        with st.expander("🚫 Missing Skills", expanded=True):
            if match_eval.missing_skills:
                for skill in match_eval.missing_skills:
                    st.write(f"- {skill}")
            else:
                st.success("No critical missing skills identified.")
                
        with st.expander("⚠️ Empty or Missing Details in Resume"):
            if empty_details:
                for detail in empty_details:
                    st.warning(detail)
            else:
                st.success("All critical details are present in the resume.")
                
        with st.expander("📄 Raw JSON Output"):
            final_output = {
                "resume_details": data.model_dump(),
                "empty_or_missing_details": empty_details,
                "match_score": match_eval.match_score,
                "reasons_for_match": match_eval.reasons_for_match,
                "reasons_for_mismatch": match_eval.reasons_for_mismatch,
                "missing_skills": match_eval.missing_skills
            }
            st.json(final_output)