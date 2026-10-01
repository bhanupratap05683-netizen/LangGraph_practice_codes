import json
import re
from typing import List

import ollama
import streamlit as st
from pydantic import BaseModel, Field, ConfigDict, ValidationError

MODEL_NAME = "qwen2.5-coder:7b"
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("❌ Warning: PyMuPDF not installed. Install with 'pip install pymupdf'")

try:
    from tqdm.asyncio import tqdm_asyncio
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️ Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")

import ollama
from pydantic import BaseModel, Field
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- CONFIGURABLE CONSTANTS ---
INPUT_FOLDER = "./resumes"          # Folder containing the 50 PDF resumes
OUTPUT_CSV = "./ranked_resumes.csv" # Output file path
MODEL_NAME = "qwen2.5-coder:7b"     # Ollama model to use
MAX_CONCURRENT_TASKS = 3            # Semaphore limit to prevent overloading local Ollama
BATCH_CHUNK_SIZE = 10               # Process resumes in chunks of 10 for better memory management

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
class ResumeData(BaseModel):
    name: str = Field(default="Unknown", description="Full name of the candidate")
    email: str = Field(default="", description="Candidate's email address")
    phone: str = Field(default="", description="Candidate's phone number")
    years_of_experience: float = Field(default=0.0, description="Total calculated work experience in years")
    skills: List[str] = Field(default_factory=list, description="All technical skills, programming languages, and analytical tools mentioned")
    education: List[str] = Field(default_factory=list, description="List of degrees, universities, or schools attended")
    last_3_job_titles: List[str] = Field(default_factory=list, description="List of recent job titles held by candidate")

class MatchEvaluation(BaseModel):
    match_score: int = Field(description="Match score between 0 and 100")
    reasons_for_match: List[str] = Field(description="List of reasons why the resume matches the job description")
    reasons_for_mismatch: List[str] = Field(description="List of reasons why the resume does not fully match the job description")
    missing_skills: List[str] = Field(description="List of specific skills mentioned in the job description but missing from the resume")


def extract_email_fallback(text: str) -> str:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else ""

def extract_phone_fallback(text: str) -> str:
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,5}", text)
    return match.group(0).strip() if match else ""

def chunk_text(text: str, max_words: int = 3000) -> str:
    """
    Text Chunking: Truncates text if it exceeds a reasonable word limit.
    Prevents context window overflow while retaining the most critical info (usually at the top of resumes).
    """
