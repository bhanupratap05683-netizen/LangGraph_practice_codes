import json
import re
from typing import List

import ollama
import streamlit as st
from pydantic import BaseModel, Field, ConfigDict, ValidationError
import ollama
from pydantic import BaseModel,


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
async def parse_resume_local_async(raw_text: str, model_name: str = MODEL_NAME) -> ResumeData:
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
       return result 
async def match_resume_to_job_async(
    resume_data: ResumeData, 
    empty_details: List[str],
    job_description: str, 
    model_name: str = MODEL_NAME
) -> MatchEvaluation:
    system_prompt = (
            "You are an expert recruitment matcher. Compare the candidate's parsed resume data with the provided job description.\n\n"
            "EVALUATION RULES:\n"
            "- MATCH SCORE: Provide an integer score from 0 to 100. Deduct points for missing critical details and missing skills.\n"
            "- REASONS FOR MATCH: Highlight specific skills, experiences, or qualifications that directly align.\n"
            "- REASONS FOR MISMATCH: Highlight areas where the candidate falls short.\n"
            "- MISSING SKILLS: List specific technical skills, tools, or qualifications explicitly mentioned in the job description but absent."
        )
    resume_json = resume_data.model_dump_json(indent=2)
        empty_details_str = ", ".join(empty_details)
                if empty_details else "None"

             async def parse_resume_local_async(raw_text: str, model_name: str = MODEL_NAME) -> ResumeData:
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
async def match_resume_to_job_async(
    resume_data: ResumeData, 
    empty_details: List[str],
    job_description: str, 
    model_name: str = MODEL_NAME
) -> MatchEvaluation:
    system_prompt = (
        "You are an expert recruitment matcher. Compare the candidate's parsed resume data with the provided job description.\n\n"
        "EVALUATION RULES:\n"
        "- MATCH SCORE: Provide an integer score from 0 to 100. Deduct points for missing critical details and missing skills.\n"
        "- REASONS FOR MATCH: Highlight specific skills, experiences, or qualifications that directly align.\n"
        "- REASONS FOR MISMATCH: Highlight areas where the candidate falls short.\n"
        "- MISSING SKILLS: List specific technical skills, tools, or qualifications explicitly mentioned in the job description but absent."
    )

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
def extract_phone_fallback(text: str) -> str:
    # Matches international formats: +91 98765 43210, (555) 234-5678, +1-555-234-5678, etc.
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,5}", text)
    return match.group(0).strip() if match else ""

    

