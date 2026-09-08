"""
RESUME BATCH PROCESSOR & RANKER
-------------------------------
Setup:
  1. Ensure Ollama is running locally (e.g., `ollama serve`).
  2. Create a folder named 'resumes' in the same directory and place your PDF resumes inside.
  3. Run the script: `python resume_ranker.py`
"""

import json
import re
import os
import asyncio
import csv
import logging
from typing import List
from pathlib import Path

# Third-party imports
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

# ----------------------------------------------------
# 0. Configuration & Setup
# ----------------------------------------------------

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
# 1. Pydantic Schemas (V2 Compatible)
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

# ----------------------------------------------------
# 2. Helper Functions (Regex, Chunking, Extraction)
# ----------------------------------------------------

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
    words = text.split()
    if len(words) > max_words:
        logging.warning("Text exceeds limit. Truncating to prevent context overflow.")
        return " ".join(words[:max_words]) + "\n\n[Note: Resume text was truncated due to length.]"
    return text

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a single PDF file using PyMuPDF."""
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF is required. Install via 'pip install pymupdf'")
    
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
    return text

def get_empty_or_missing_details(data: ResumeData) -> List[str]:
    missing = []
    if not data.name or data.name == "Unknown":
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
# 3. Async LLM Processing Functions
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
        format=ResumeData.model_json_schema(),
        options={"temperature": 0.0, "num_ctx": 4096},
    )

    result = ResumeData.model_validate_json(response.message.content)

    # Fallback checks
    if not result.email:
        result.email = extract_email_fallback(raw_text)
    if not result.phone:
        result.phone = extract_phone_fallback(raw_text)

    result.last_3_job_titles = result.last_3_job_titles[:3]
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

# ----------------------------------------------------
# 4. Batch Processing & Chunking Logic
# ----------------------------------------------------

async def process_single_resume(pdf_path: str, semaphore: asyncio.Semaphore) -> dict:
    """Processes a single resume with concurrency control."""
    async with semaphore:
        file_name = os.path.basename(pdf_path)
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            
            if not raw_text.strip():
                raise ValueError("No text could be extracted from the PDF.")
            
            # Apply text chunking/truncation for safety
            safe_text = chunk_text(raw_text, max_words=3000)
            
            # 1. Parse Resume
            data = await parse_resume_local_async(safe_text)
            
            # 2. Identify missing details
            empty_details = get_empty_or_missing_details(data)
            
            # 3. Evaluate Match
            match_eval = await match_resume_to_job_async(data, empty_details, TARGET_JOB_DESCRIPTION)
            
            return {
                "file_name": file_name,
                "name": data.name,
                "email": data.email,
                "phone": data.phone,
                "years_of_experience": data.years_of_experience,
                "match_score": match_eval.match_score,
                "skills": ", ".join(data.skills),
                "reasons_for_match": " | ".join(match_eval.reasons_for_match),
                "reasons_for_mismatch": " | ".join(match_eval.reasons_for_mismatch),
                "missing_skills": ", ".join(match_eval.missing_skills),
                "status": "Success"
            }
        except Exception as e:
            logging.error(f"Failed to process {file_name}: {e}")
            return {
                "file_name": file_name,
                "name": "Error",
                "email": "",
                "phone": "",
                "years_of_experience": 0,
                "match_score": 0,
                "skills": "",
                "reasons_for_match": "",
                "reasons_for_mismatch": str(e),
                "missing_skills": "",
                "status": "Failed"
            }

def chunk_list(lst: list, chunk_size: int):
    """Yield successive chunk_size-sized chunks from lst (Batch Chunking)."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

async def main_async():
    # 1. Validate Input Folder
    if not os.path.isdir(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        logging.error(f"Input folder '{INPUT_FOLDER}' was missing. It has been created. Please add PDF resumes and run again.")
        return

    # 2. Gather all PDF files
    pdf_files = [str(p) for p in Path(INPUT_FOLDER).rglob("*.pdf")]
    if not pdf_files:
        logging.warning(f"No PDF files found in '{INPUT_FOLDER}'.")
        return

    logging.info(f"✅ Found {len(pdf_files)} PDF resumes. Starting batch processing...")

    # 3. Setup Concurrency Control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    all_results = []

    # 4. Process in Chunks (Batch Chunking for memory/queue management)
    for chunk in chunk_list(pdf_files, BATCH_CHUNK_SIZE):
        logging.info(f"Processing chunk of {len(chunk)} resumes...")
        
        tasks = [process_single_resume(pdf_path, semaphore) for pdf_path in chunk]
        
        # Progress bar for the current chunk
        if HAS_TQDM:
            chunk_results = await tqdm_asyncio.gather(*tasks, desc="Processing Resumes")
        else:
            chunk_results = await asyncio.gather(*tasks)
            
        all_results.extend(chunk_results)

    # 5. Filter, Sort, and Rank
    # Sort ALL by match_score descending (failed ones will naturally fall to the bottom with score 0)
    ranked_results = sorted(all_results, key=lambda x: x["match_score"], reverse=True)

    # Add Rank column
    for idx, res in enumerate(ranked_results, start=1):
        res["rank"] = idx

    # 6. Output Top 10 to Console
    print("\n" + "="*90)
    print("🏆 TOP 10 RANKED CANDIDATES 🏆")
    print("="*90)
    for res in ranked_results[:10]:
        if res["status"] == "Success":
            print(f"Rank #{res['rank']:<2} | Score: {res['match_score']:<3}/100 | Name: {res['name']:<20} | File: {res['file_name']}")
            print(f"  ↳ Skills: {res['skills']}")
            print(f"  ↳ Match:  {res['reasons_for_match'][:120]}...")
            print("-" * 90)

    # 7. Export Full Results to CSV
    csv_columns = [
        "rank", "file_name", "name", "email", "phone", "years_of_experience", 
        "match_score", "skills", "reasons_for_match", "reasons_for_mismatch", 
        "missing_skills", "status"
    ]
    
    try:
        with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(ranked_results)
        logging.info(f"✅ Successfully exported all {len(ranked_results)} results to '{OUTPUT_CSV}'")
    except Exception as e:
        logging.error(f"Failed to write CSV: {e}")

if __name__ == "__main__":
    asyncio.run(main_async())