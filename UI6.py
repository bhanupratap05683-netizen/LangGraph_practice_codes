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
