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