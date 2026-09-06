# ============================================================
# app.py
# Local Resume Matcher
# Streamlit + Ollama + Qwen2.5-Coder 7B + Pydantic
# ============================================================

import json
import re
from typing import List

import ollama
import streamlit as st
from pydantic import BaseModel, Field, ConfigDict, ValidationError


# ============================================================
# 0. Configuration
# ============================================================

MODEL_NAME = "qwen2.5-coder:7b"

TARGET_JOB_DESCRIPTION = """
Job Title: Financial Data Analyst (Quantitative Analytics)
Department: Quantitative Research & Financial Analytics
Employment Type: Full-Time

Role Overview:
We are looking for a rigorous, mathematically minded Financial Data Analyst
to bridge the gap between complex financial datasets and strategic
decision-making. In this role, you will design statistical models,
analyze market/transactional data, and build reproducible quantitative
frameworks to evaluate risk, forecast performance, and optimize portfolio
returns.

Key Responsibilities:
- Quantitative Modeling & Forecasting: Develop, backtest, and refine
  predictive statistical models, time-series forecasts (ARIMA, GARCH),
  and risk metrics (VaR, stress-testing, Sharpe ratio).
- Financial Data Engineering: Ingest, clean, and normalize structured
  and unstructured financial data (tick data, balance sheets,
  macroeconomic indicators, order books) across databases and API feeds.
- SQL & Data Extraction: Write high-performance SQL queries involving
  window functions, indexing, and CTEs.
- Advanced Financial Modeling: Build robust financial models and
  automated reporting systems in Excel utilizing Power Query,
  VBA/macros, and dynamic matrix formulas.
- Algorithmic Scripting: Implement data processing pipelines and
  exploratory quantitative analyses in Python (Pandas, NumPy, SciPy,
  statsmodels) or R.
- Performance & Risk Dashboards: Design and maintain automated risk
  and performance attribution dashboards in Power BI, Tableau,
  or Dash/Streamlit.
- Model Validation & Integrity: Ensure data hygiene, investigate
  pricing or accounting anomalies, and validate statistical assumptions.

Required Qualifications:
- Education: Bachelor's or Master's degree in Finance, Economics,
  Quantitative Finance, Applied Mathematics, Statistics, Computer Science,
  or a related discipline.
- Core Technical Stack: Python or R (pandas, numpy, statsmodels, scipy),
  Advanced SQL, Advanced Excel.
- Financial Acumen: Corporate finance, valuation methods, capital markets,
  derivatives, and fixed-income/equity instruments.
- Statistical Foundations: Probability, linear regression, multivariate
  analysis, hypothesis testing, and time-series analysis.

Preferred Qualifications:
- CFA or FRM progress/completion.
- Market data terminals and APIs.
- Financial machine learning or factor modeling.
- Cloud data warehouses and Git/GitHub.
"""


# ============================================================
# 1. Pydantic Schemas
# ============================================================

class ResumeData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    email: str = ""
    phone: str = ""

    years_of_experience: float = Field(
        default=0.0,
        ge=0,
        description="Total calculated work experience in years"
    )

    skills: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    last_3_job_titles: List[str] = Field(default_factory=list)


class MatchEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    match_score: int = Field(
        default=0,
        ge=0,
        le=100
    )

    reasons_for_match: List[str] = Field(default_factory=list)
    reasons_for_mismatch: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)


# ============================================================
# 2. PDF Text Extraction
# ============================================================

try:
    import fitz

    HAS_PYMUPDF = True

except ImportError:
    HAS_PYMUPDF = False


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from a PDF uploaded through Streamlit."""

    if not HAS_PYMUPDF:
        raise ImportError(
            "PyMuPDF is required. Install it using: pip install pymupdf"
        )

    text = ""

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

    return text


# ============================================================
# 3. Regex Fallback Helpers
# ============================================================

def extract_email_fallback(text: str) -> str:
    match = re.search(
        r"[\w\.-]+@[\w\.-]+\.\w+",
        text
    )

    return match.group(0) if match else ""


def extract_phone_fallback(text: str) -> str:
    match = re.search(
        r"(\+?\d{1,3}[-.\s]?)?"
        r"(\(?\d{3,5}\)?[-.\s]?)?"
        r"\d{3,5}[-.\s]?\d{4,5}",
        text
    )

    return match.group(0).strip() if match else ""


# ============================================================
# 4. Ollama Connection Checking
# ============================================================

def check_ollama_connection(model_name: str) -> tuple[bool, str]:
    """
    Check whether Ollama is running and whether the requested model
    is available locally.
    """

    try:
        models_response = ollama.list()

        # Ollama Python SDK versions may return either a dictionary
        # or an object-like response.
        if hasattr(models_response, "models"):
            models = models_response.models
        else:
            models = models_response.get("models", [])

        installed_models = []

        for model in models:
            if isinstance(model, dict):
                model_name_value = model.get("name", "")
            else:
                model_name_value = getattr(model, "name", "")

            if model_name_value:
                installed_models.append(model_name_value)

        if model_name not in installed_models:

            return False, (
                f"Model '{model_name}' is not installed.\n\n"
                f"Installed models: {installed_models}\n\n"
                f"Run: ollama pull {model_name}"
            )

        return True, f"Ollama is ready. Using model: {model_name}"

    except Exception as e:

        return False, (
            "Could not connect to Ollama.\n\n"
            "Make sure Ollama is running.\n\n"
            f"Error: {e}"
        )


# ============================================================
# 5. Safe Ollama JSON Helper
# ============================================================

def call_ollama_json(
    model_name: str,
    messages: list,
    schema: dict
) -> str:
    """
    Call Ollama and return the model's JSON response.

    The model is instructed to return structured JSON.
    Pydantic validates the final response.
    """

    try:

        response = ollama.chat(
            model=model_name,
            messages=messages,
            format=schema,
            options={
                "temperature": 0.0,
                "num_ctx": 4096,
            }
        )

        content = response.message.content

        if not content:
            raise ValueError("Ollama returned an empty response.")

        return content

    except Exception as e:

        raise RuntimeError(
            f"Ollama request failed: {e}"
        )


# ============================================================
# 6. Resume Extraction
# ============================================================

def parse_resume_local(
    raw_text: str,
    model_name: str = MODEL_NAME
) -> ResumeData:

    system_prompt = """
You are an expert recruitment resume parser.

Your task is to extract factual information from the resume.

IMPORTANT RULES:

1. Return ONLY valid JSON matching the provided schema.
2. Do not return markdown.
3. Do not return explanations.
4. Do not invent information.
5. If a field is missing, return:
   - "" for missing strings
   - [] for missing lists
   - 0 for missing experience
6. Extract the candidate's actual name.
7. Extract the email and phone exactly as printed.
8. Extract technical skills, programming languages, software,
   financial tools, databases, and analytical tools.
9. Extract education as degree/institution strings.
10. Extract the three most recent job titles.
11. Calculate total work experience in years.
12. Do not double-count overlapping employment periods.
13. If dates are unclear, make a conservative estimate.
14. Do not treat internships as full-time experience unless
    the resume clearly indicates that they should count.
15. Do not infer skills merely because they are mentioned
    in the job description.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": (
                "Extract all details from this resume.\n\n"
                "RESUME TEXT:\n"
                f"{raw_text}"
            )
        }
    ]

    try:

        content = call_ollama_json(
            model_name,
            messages,
            ResumeData.model_json_schema()
        )

        result = ResumeData.model_validate_json(content)

    except ValidationError as e:

        raise RuntimeError(
            f"Resume JSON validation failed: {e}"
        )

    except Exception as e:

        raise RuntimeError(
            f"Resume extraction failed: {e}"
        )

    # Regex fallback
    if not result.email:
        result.email = extract_email_fallback(raw_text)

    if not result.phone:
        result.phone = extract_phone_fallback(raw_text)

    # Keep only three recent titles
    result.last_3_job_titles = result.last_3_job_titles[:3]

    return result


# ============================================================
# 7. Missing Details Detection
# ============================================================

def get_empty_or_missing_details(
    data: ResumeData
) -> List[str]:

    missing = []

    if not data.name or not data.name.strip():
        missing.append("Candidate name is missing")

    if not data.email:
        missing.append("Email address is missing")

    if not data.phone:
        missing.append("Phone number is missing")

    if data.years_of_experience <= 0:
        missing.append(
            "Years of experience is missing or zero"
        )

    if not data.skills:
        missing.append("No technical skills listed")

    if not data.education:
        missing.append("Education details are missing")

    if not data.last_3_job_titles:
        missing.append("Job titles are missing")

    return missing


# ============================================================
# 8. Skill Normalization
# ============================================================

def normalize_skill(skill: str) -> str:
    """Normalize a skill for comparison."""

    return re.sub(
        r"[^a-z0-9+#.]",
        "",
        skill.lower()
    )


def skill_matches(
    required_skill: str,
    candidate_skills: List[str]
) -> bool:

    required = normalize_skill(required_skill)

    for candidate_skill in candidate_skills:

        candidate = normalize_skill(candidate_skill)

        if required == candidate:
            return True

        if required in candidate or candidate in required:
            return True

    return False


# ============================================================
# 9. Transparent Weighted Scoring
# ============================================================

def calculate_weighted_score(
    resume_data: ResumeData,
    missing_skills: List[str],
    job_description: str
) -> int:
    """
    Calculate a transparent score.

    This is NOT a final hiring decision.
    It is a screening score based on available resume evidence.

    Weights:
    - Technical skills: 40
    - Experience: 25
    - Education: 15
    - Financial/quantitative alignment: 15
    - Resume completeness: 5
    """

    score = 0

    # --------------------------------------------------------
    # Technical Skills: 40 points
    # --------------------------------------------------------

    required_skills = [
        "Python",
        "SQL",
        "Excel",
        "Pandas",
        "NumPy",
        "SciPy",
        "statsmodels",
        "Power BI",
        "Tableau",
        "Streamlit",
        "R",
        "VBA",
        "Power Query",
        "Git",
    ]

    matched_skills = 0

    for skill in required_skills:

        if skill_matches(skill, resume_data.skills):
            matched_skills += 1

    technical_score = (
        matched_skills / len(required_skills)
    ) * 40

    score += technical_score

    # --------------------------------------------------------
    # Experience: 25 points
    # --------------------------------------------------------

    experience = resume_data.years_of_experience

    if experience >= 5:
        experience_score = 25

    elif experience >= 3:
        experience_score = 20

    elif experience >= 1:
        experience_score = 12

    elif experience > 0:
        experience_score = 5

    else:
        experience_score = 0

    score += experience_score

    # --------------------------------------------------------
    # Education: 15 points
    # --------------------------------------------------------

    education_text = " ".join(
        resume_data.education
    ).lower()

    education_keywords = [
        "finance",
        "economics",
        "quantitative",
        "mathematics",
        "statistics",
        "computer science",
        "business",
        "data",
    ]

    education_matches = sum(
        1
        for keyword in education_keywords
        if keyword in education_text
    )

    if education_matches >= 2:
        education_score = 15

    elif education_matches == 1:
        education_score = 10

    elif resume_data.education:
        education_score = 5

    else:
        education_score = 0

    score += education_score

    # --------------------------------------------------------
    # Financial / Quantitative Alignment: 15 points
    # --------------------------------------------------------

    resume_text = (
        " ".join(resume_data.skills)
        + " "
        + " ".join(resume_data.education)
        + " "
        + " ".join(resume_data.last_3_job_titles)
    ).lower()

    finance_keywords = [
        "finance",
        "financial",
        "investment",
        "portfolio",
        "risk",
        "valuation",
        "statistics",
        "analytics",
        "quantitative",
        "accounting",
        "economics",
    ]

    finance_matches = sum(
        1
        for keyword in finance_keywords
        if keyword in resume_text
    )

    finance_score = min(
        15,
        finance_matches * 2.5
    )

    score += finance_score

    # --------------------------------------------------------
    # Resume Completeness: 5 points
    # --------------------------------------------------------

    completeness_score = 0

    if resume_data.name:
        completeness_score += 1

    if resume_data.email:
        completeness_score += 1

    if resume_data.phone:
        completeness_score += 1

    if resume_data.education:
        completeness_score += 1

    if resume_data.skills:
        completeness_score += 1

    score += completeness_score

    return max(0, min(100, round(score)))


# ============================================================
# 10. Resume Matching
# ============================================================

def match_resume_to_job(
    resume_data: ResumeData,
    empty_details: List[str],
    job_description: str,
    model_name: str = MODEL_NAME
) -> MatchEvaluation:

    system_prompt = """
You are an expert recruitment matcher.

Compare the candidate's resume data with the job description.

IMPORTANT RULES:

1. Return ONLY valid JSON matching the provided schema.
2. Do not return markdown.
3. Do not return explanations outside JSON.
4. Do not invent skills or experience.
5. Match score must be an integer from 0 to 100.
6. Give specific reasons for matching.
7. Give specific reasons for mismatch.
8. List only skills required by the JD that are absent
   from the resume.
9. Do not list every JD skill automatically.
10. Distinguish between:
    - explicitly demonstrated skills
    - partially demonstrated skills
    - missing skills
11. Missing information should reduce confidence.
12. Do not make a hiring decision.
13. Do not use protected characteristics.
14. Do not infer age, gender, religion, race, disability,
    marital status, or other sensitive attributes.
15. Focus only on job-related evidence.
"""

    resume_json = resume_data.model_dump_json(indent=2)

    empty_details_str = (
        ", ".join(empty_details)
        if empty_details
        else "None"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": (
                "RESUME DATA:\n"
                f"{resume_json}\n\n"
                "EMPTY / MISSING DETAILS:\n"
                f"{empty_details_str}\n\n"
                "JOB DESCRIPTION:\n"
                f"{job_description}"
            )
        }
    ]

    try:

        content = call_ollama_json(
            model_name,
            messages,
            MatchEvaluation.model_json_schema()
        )

        result = MatchEvaluation.model_validate_json(content)

        # Safety constraint
        result.match_score = max(
            0,
            min(100, result.match_score)
        )

        return result

    except ValidationError as e:

        raise RuntimeError(
            f"Matching JSON validation failed: {e}"
        )

    except Exception as e:

        raise RuntimeError(
            f"Resume matching failed: {e}"
        )


# ============================================================
# 11. Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume to Job Description Matcher")

st.markdown(
    "Target Role: **Financial Data Analyst "
    "(Quantitative Analytics)**"
)

st.divider()


# ============================================================
# 12. Ollama Status
# ============================================================

with st.sidebar:

    st.header("⚙️ Local AI Settings")

    st.write(f"**Model:** `{MODEL_NAME}`")

    if st.button("🔌 Check Ollama Connection"):

        with st.spinner("Checking Ollama..."):

            connected, message = check_ollama_connection(
                MODEL_NAME
            )

        if connected:
            st.success(message)

        else:
            st.error(message)

    st.info(
        "This application uses your locally installed "
        "Ollama model. No resume data is sent to a "
        "cloud LLM by this code."
    )


# ============================================================
# 13. Input Section
# ============================================================

col1, col2 = st.columns([1, 1])

with col1:

    input_method = st.radio(
        "How would you like to provide the resume?",
        ("Upload PDF", "Paste Text")
    )

raw_text = ""

with col2:

    if input_method == "Upload PDF":

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"]
        )

        if uploaded_file is not None:

            try:

                raw_text = extract_text_from_pdf_bytes(
                    uploaded_file.getvalue()
                )

                if not raw_text.strip():

                    st.warning(
                        "No text was extracted from this PDF. "
                        "It may be scanned or image-based."
                    )

            except Exception as e:

                st.error(f"PDF extraction failed: {e}")

    else:

        raw_text = st.text_area(
            "Paste resume text here:",
            height=250,
            placeholder="Paste the candidate's resume text..."
        )


# ============================================================
# 14. Process Button
# ============================================================

if st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True
):

    if not raw_text.strip():

        st.error(
            "Please provide a resume "
            "(either upload a PDF or paste text)."
        )

    else:

        # ----------------------------------------------------
        # Check Ollama
        # ----------------------------------------------------

        with st.spinner("🔌 Checking Ollama connection..."):

            connected, connection_message = (
                check_ollama_connection(MODEL_NAME)
            )

        if not connected:

            st.error(connection_message)

            st.stop()

        # ----------------------------------------------------
        # Parse Resume
        # ----------------------------------------------------

        try:

            with st.spinner(
                "🤖 Extracting and parsing resume data..."
            ):

                data = parse_resume_local(
                    raw_text,
                    MODEL_NAME
                )

        except Exception as e:

            st.error(f"❌ Resume extraction failed: {e}")

            st.stop()

        # ----------------------------------------------------
        # Missing Details
        # ----------------------------------------------------

        empty_details = get_empty_or_missing_details(data)

        # ----------------------------------------------------
        # Match Evaluation
        # ----------------------------------------------------

        try:

            with st.spinner(
                "📊 Evaluating match against Financial Data Analyst role..."
            ):

                match_eval = match_resume_to_job(
                    data,
                    empty_details,
                    TARGET_JOB_DESCRIPTION,
                    MODEL_NAME
                )

        except Exception as e:

            st.error(f"❌ Resume matching failed: {e}")

            st.stop()

        # ----------------------------------------------------
        # Transparent Weighted Score
        # ----------------------------------------------------

        weighted_score = calculate_weighted_score(
            data,
            match_eval.missing_skills,
            TARGET_JOB_DESCRIPTION
        )

        # ----------------------------------------------------
        # Display Results
        # ----------------------------------------------------

        st.success("✅ Analysis Complete!")

        st.divider()

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "AI Match Score",
            f"{match_eval.match_score}/100"
        )

        m2.metric(
            "Weighted Score",
            f"{weighted_score}/100"
        )

        m3.metric(
            "Experience",
            f"{data.years_of_experience} Years"
        )

        m4.metric(
            "Skills Identified",
            len(data.skills)
        )

        st.subheader(
            f"👤 Candidate: {data.name or 'Not specified'}"
        )

        st.write(
            f"📧 {data.email or 'Not specified'}  |  "
            f"📞 {data.phone or 'Not specified'}"
        )

        st.write(
            f"🎓 **Education:** "
            f"{', '.join(data.education) if data.education else 'Not specified'}"
        )

        st.write(
            f"💼 **Recent Titles:** "
            f"{', '.join(data.last_3_job_titles) if data.last_3_job_titles else 'Not specified'}"
        )

        st.divider()

        # ----------------------------------------------------
        # Reasons for Match
        # ----------------------------------------------------

        with st.expander(
            "✅ Reasons for Match",
            expanded=True
        ):

            if match_eval.reasons_for_match:

                for reason in match_eval.reasons_for_match:

                    st.write(f"- {reason}")

            else:

                st.info(
                    "No specific matching reasons identified."
                )

        # ----------------------------------------------------
        # Reasons for Mismatch
        # ----------------------------------------------------

        with st.expander(
            "❌ Reasons for Mismatch",
            expanded=True
        ):

            if match_eval.reasons_for_mismatch:

                for reason in match_eval.reasons_for_mismatch:

                    st.write(f"- {reason}")

            else:

                st.info(
                    "No mismatches identified."
                )

        # ----------------------------------------------------
        # Missing Skills
        # ----------------------------------------------------

        with st.expander(
            "🚫 Missing Skills",
            expanded=True
        ):

            if match_eval.missing_skills:

                for skill in match_eval.missing_skills:

                    st.write(f"- {skill}")

            else:

                st.success(
                    "No critical missing skills identified."
                )

        # ----------------------------------------------------
        # Empty Details
        # ----------------------------------------------------

        with st.expander(
            "⚠️ Empty or Missing Details in Resume"
        ):

            if empty_details:

                for detail in empty_details:

                    st.warning(detail)

            else:

                st.success(
                    "All critical details are present in the resume."
                )

        # ----------------------------------------------------
        # Score Explanation
        # ----------------------------------------------------

        with st.expander(
            "📊 Weighted Score Explanation"
        ):

            st.write(
                "The weighted score is calculated from "
                "resume evidence, not generated entirely "
                "by the LLM."
            )

            st.write(
                "- Technical Skills: **40 points**"
            )

            st.write(
                "- Experience: **25 points**"
            )

            st.write(
                "- Education: **15 points**"
            )

            st.write(
                "- Financial/Quantitative Alignment: **15 points**"
            )

            st.write(
                "- Resume Completeness: **5 points**"
            )

            st.info(
                "This score is a screening aid, not a final "
                "hiring decision."
            )

        # ----------------------------------------------------
        # Raw JSON Output
        # ----------------------------------------------------

        with st.expander("📄 Raw JSON Output"):

            final_output = {
                "resume_details": data.model_dump(),
                "empty_or_missing_details": empty_details,
                "ai_match_score": match_eval.match_score,
                "weighted_score": weighted_score,
                "reasons_for_match": match_eval.reasons_for_match,
                "reasons_for_mismatch": match_eval.reasons_for_mismatch,
                "missing_skills": match_eval.missing_skills
            }

            st.json(final_output)