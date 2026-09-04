import json
import re
import sys
import os
import asyncio
from typing import List
import ollama
from pydantic import BaseModel, Field

# ----------------------------------------------------
# 0. Hardcoded Target Job Description (NEW ADDITION)
# ----------------------------------------------------

TARGET_JOB_DESCRIPTION = """
Job Title: Financial Data Analyst (Quantitative Analytics)
Department: Quantitative Research & Financial Analytics
Employment Type: Full-Time
Location: [Remote / Hybrid / On-site — City, Country]

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
# 1. Non-Nullable Schema (Forces the model to populate)
# ----------------------------------------------------

class ResumeData(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(
        description="Candidate's email address (use empty string '' if not found)"
    )
    phone: str = Field(
        description="Candidate's phone number with country/area code (use empty string '' if not found)"
    )
    years_of_experience: float = Field(
        description="Total calculated work experience in years (e.g. 4.0)"
    )
    skills: List[str] = Field(
        description="All technical skills, programming languages, and analytical tools mentioned"
    )
    education: List[str] = Field(
        description="List of degrees, universities, or schools attended (e.g. ['B.Com - Bangalore University'])"
    )
    last_3_job_titles: List[str] = Field(
        description="List of job titles held by candidate"
    )


# ----------------------------------------------------
# 2. Regex Fallback Helpers (Guarantees Email & Phone)
# ----------------------------------------------------

def extract_email_fallback(text: str) -> str:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else ""


def extract_phone_fallback(text: str) -> str:
    # Matches international formats: +91 98765 43210, (555) 234-5678, +1-555-234-5678, etc.
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,5}", text)
    return match.group(0).strip() if match else ""


# ----------------------------------------------------
# 3. PDF Text Extraction (Single PDF)
# ----------------------------------------------------

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Install with 'pip install pymupdf' to enable PDF handling.")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a single PDF file using PyMuPDF."""
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF is required for PDF extraction. Install it via 'pip install pymupdf'")
    
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text


# ----------------------------------------------------
# 4. Main Extraction Function (ASYNC UPDATED)
# ----------------------------------------------------

async def parse_resume_local_async(
    raw_text: str, model_name: str = "qwen2.5-coder:7b"
) -> ResumeData:
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

    # Using asyncio.to_thread to make the synchronous ollama.chat call non-blocking
    response = await asyncio.to_thread(
        ollama.chat,
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Extract all details from this resume text:\n\n{raw_text}",
            },
        ],
        format=ResumeData.model_json_schema(),
        options={
            "temperature": 0.0,
            "num_ctx": 4096,
        },
    )

    result = ResumeData.model_validate_json(response.message.content)

    # Fallback checks: If the LLM returned empty strings for contact info, fill via regex
    if not result.email:
        result.email = extract_email_fallback(raw_text)
    if not result.phone:
        result.phone = extract_phone_fallback(raw_text)

    # Ensure max 3 titles cleanly
    result.last_3_job_titles = result.last_3_job_titles[:3]

    return result


# ----------------------------------------------------
# 5. Match Evaluation Schema
# ----------------------------------------------------

class MatchEvaluation(BaseModel):
    match_score: int = Field(description="Match score between 0 and 100")
    reasons_for_match: List[str] = Field(description="List of reasons why the resume matches the job description")
    reasons_for_mismatch: List[str] = Field(description="List of reasons why the resume does not fully match the job description")
    missing_skills: List[str] = Field(description="List of specific skills mentioned in the job description but missing from the resume")


# ----------------------------------------------------
# 6. Helper to Identify Empty Details
# ----------------------------------------------------

def get_empty_or_missing_details(data: ResumeData) -> List[str]:
    """Programmatically identifies empty or missing fields for 100% accuracy."""
    missing = []
    if not data.name or data.name.strip() == "":
        missing.append("Candidate name is missing")
    if not data.email:
        missing.append("Email address is missing")
    if not data.phone:
        missing.append("Phone number is missing")
    if data.years_of_experience <= 0:
        missing.append("Years of experience is missing or zero")
    if not data.skills:
        missing.append("No technical skills listed")
    if not data.education:
        missing.append("Education details are missing")
    if not data.last_3_job_titles:
        missing.append("Job titles are missing")
    return missing


# ----------------------------------------------------
# 7. Resume-to-Job Matcher Function (ASYNC UPDATED)
# ----------------------------------------------------

async def match_resume_to_job_async(
    resume_data: ResumeData, 
    empty_details: List[str],
    job_description: str, 
    model_name: str = "qwen2.5-coder:7b"
) -> MatchEvaluation:
    system_prompt = (
        "You are an expert recruitment matcher. Compare the candidate's parsed resume data with the provided job description.\n\n"
        "EVALUATION RULES:\n"
        "- MATCH SCORE: Provide an integer score from 0 to 100. Deduct points for missing critical details (listed below) and missing skills.\n"
        "- REASONS FOR MATCH: Highlight specific skills, experiences, or qualifications from the resume that directly align with the job description.\n"
        "- REASONS FOR MISMATCH: Highlight areas where the candidate falls short (e.g., insufficient years of experience, wrong domain, missing contact details, missing soft skills).\n"
        "- MISSING SKILLS: List specific technical skills, tools, or qualifications explicitly mentioned in the job description but absent from the resume."
    )

    resume_json = resume_data.model_dump_json(indent=2)
    empty_details_str = ", ".join(empty_details) if empty_details else "None"

    # Using asyncio.to_thread to make the synchronous ollama.chat call non-blocking
    response = await asyncio.to_thread(
        ollama.chat,
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Resume Data:\n{resume_json}\n\nEmpty/Missing Resume Details:\n{empty_details_str}\n\nJob Description:\n{job_description}",
            },
        ],
        format=MatchEvaluation.model_json_schema(),
        options={
            "temperature": 0.0,
            "num_ctx": 4096,
        },
    )

    return MatchEvaluation.model_validate_json(response.message.content)


# ----------------------------------------------------
# 8. Main Async Execution Block
# ----------------------------------------------------

async def main_async():
    raw_text = ""
    
    # Check if a single PDF file is passed as a command-line argument
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isfile(target) and target.lower().endswith(".pdf"):
            print(f"Processing single PDF: {target}\n")
            raw_text = extract_text_from_pdf(target)
            
            if not raw_text.strip():
                print("No text could be extracted from the PDF. It might be image-only or corrupted.")
                return
        else:
            print("Invalid argument. Please provide a valid single PDF file.")
            print("Usage: python script.py path/to/resume.pdf")
            return
    else:
        # ORIGINAL FUNCTIONALITY: Terminal Text Input
        print("Please paste the resume text below.")
        print("When you are finished, type 'DONE' on a new line and press Enter.")
        print("(Tip: To process a single PDF, run: python script.py path/to/resume.pdf)\n")
        
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "DONE":
                    break
                lines.append(line)
            except EOFError:
                break
        
        raw_text = "\n".join(lines)
        
        if not raw_text.strip():
            print("No resume text provided. Exiting.")
            return

    print("\nExtracting and parsing data...\n")
    
    # 1. Parse Resume (Async)
    data = await parse_resume_local_async(raw_text)
    
    # 2. Identify empty details programmatically
    empty_details = get_empty_or_missing_details(data)
    
    print("Evaluating match against Financial Data Analyst (Quantitative Analytics) role...\n")
    
    # 3. Evaluate Match (Async)
    match_eval = await match_resume_to_job_async(data, empty_details, TARGET_JOB_DESCRIPTION)
    
    # 4. Build Final Unified JSON Output
    final_output = {
        "resume_details": data.model_dump(),
        "empty_or_missing_details": empty_details,
        "match_score": match_eval.match_score,
        "reasons_for_match": match_eval.reasons_for_match,
        "reasons_for_mismatch": match_eval.reasons_for_mismatch,
        "missing_skills": match_eval.missing_skills
    }
    
    print("\n" + "="*60)
    print("FINAL JSON OUTPUT")
    print("="*60)
    print(json.dumps(final_output, indent=2))


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main_async())