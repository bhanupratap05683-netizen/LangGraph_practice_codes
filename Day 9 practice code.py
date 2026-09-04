import json
import re
import sys
import os
import glob
from typing import List
import ollama
from pydantic import BaseModel, Field

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
# 3. PDF Text Extraction (New Addition)
# ----------------------------------------------------

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Install with 'pip install pymupdf' to enable PDF handling.")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
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
# 4. Main Extraction Function
# ----------------------------------------------------

def parse_resume_local(
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

    response = ollama.chat(
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
# 5. Batch Testing Function for 15 Real PDFs (New Addition)
# ----------------------------------------------------

def batch_test_pdf_resumes(directory: str = "./test_resumes"):
    """
    Tests the parsing pipeline on real PDF resumes in the specified directory.
    Chain: PDF upload (local path) → text extraction → AI parsing → structured JSON.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        print("Please download 15 real PDF resumes from Google and place them in a folder named 'test_resumes'.")
        return
    
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{directory}'.")
        return
        
    print(f"Found {len(pdf_files)} PDF(s). Processing...\n")
    
    results = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}")
        try:
            # 1. Text Extraction
            raw_text = extract_text_from_pdf(pdf_path)
            if not raw_text.strip():
                print(f"  -> Skipped: No text extracted.")
                continue
            
            # 2. AI Parsing
            parsed_data = parse_resume_local(raw_text)
            
            # 3. Structured JSON accumulation
            results.append({
                "file": filename,
                "parsed_data": parsed_data.model_dump()
            })
            print(f"  -> Success: Extracted '{parsed_data.name}'")
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")
            
    # Save all results to a single JSON file for review
    output_file = "pdf_parsing_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nTesting complete. Results saved to {output_file}")


# ----------------------------------------------------
# 6. Main Execution (Terminal Input + PDF CLI Support)
# ----------------------------------------------------
if __name__ == "__main__":
    # NEW: Check if a PDF file or directory is passed as a command-line argument
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isdir(target):
            print(f"Running batch test on directory: {target}")
            batch_test_pdf_resumes(target)
        elif os.path.isfile(target) and target.lower().endswith(".pdf"):
            print(f"Processing single PDF: {target}")
            raw_text = extract_text_from_pdf(target)
            if raw_text.strip():
                data = parse_resume_local(raw_text)
                print(data.model_dump_json(indent=2))
            else:
                print("No text could be extracted from the PDF.")
        else:
            print("Invalid argument. Please provide a valid PDF file or directory.")
    else:
        # ORIGINAL FUNCTIONALITY: Terminal Input
        print("Please paste the resume text below.")
        print("When you are finished, type 'DONE' on a new line and press Enter.")
        print("(Tip: To process a PDF, run: python script.py path/to/resume.pdf)")
        print("(Tip: To test 15 PDFs, run: python script.py ./test_resumes)\n")
        
        lines = []
        while True:
            try:
                line = input()
                # Check if the user typed the termination keyword
                if line.strip() == "DONE":
                    break
                lines.append(line)
            except EOFError:
                # Handles Ctrl+D (Mac/Linux) or Ctrl+Z (Windows) gracefully
                break
        
        raw_text = "\n".join(lines)
        
        if not raw_text.strip():
            print("No resume text provided. Exiting.")
        else:
            print("\nProcessing resume...\n")
            data = parse_resume_local(raw_text)
            print(data.model_dump_json(indent=2))