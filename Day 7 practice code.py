import json
import re
import sys
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
# 3. Main Extraction Function
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
# 4. Main Execution (Terminal Input)
# ----------------------------------------------------
if __name__ == "__main__":
    print("Please paste the resume text below.")
    print("When you are finished, type 'DONE' on a new line and press Enter.\n")
    
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