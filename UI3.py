import json
import re
from typing import List
import ollama
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

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

def parse_resume_local(raw_text: str, model_name: str = "qwen2.5-coder:7b") -> ResumeData:
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
# 4. Streamlit UI
# ----------------------------------------------------

st.set_page_config(page_title="AI Resume Parser", page_icon="📄", layout="wide")

st.title("📄 AI Resume Parser")
st.write("Extract structured data from resumes using local LLMs via Ollama.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.text_input("Ollama Model Name", value="qwen2.5-coder:7b")
    st.caption("Ensure Ollama is running and the model is pulled (`ollama pull qwen2.5-coder:7b`).")
    st.divider()
    st.markdown("### How to use:")
    st.markdown("1. Paste resume text below, OR upload a `.txt` file.")
    st.markdown("2. Click **Parse Resume**.")
    st.markdown("3. View the structured output.")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    raw_text = st.text_area(
        "Paste Resume Text Here", 
        height=300, 
        placeholder="Paste the resume text here..."
    )
    
    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8")
        st.info("File loaded successfully! Text preview is shown above.")

with col2:
    st.subheader("Output")
    
    if st.button("Parse Resume", type="primary", use_container_width=True):
        if not raw_text.strip():
            st.warning("Please provide some resume text or upload a file first.")
        else:
            with st.spinner("Parsing resume... This may take a moment depending on your local hardware."):
                try:
                    data = parse_resume_local(raw_text, model_name=model_name)
                    st.success("✅ Successfully parsed!")
                    
                    # Display results nicely
                    st.markdown(f"### 👤 {data.name}")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**📧 Email:** `{data.email or 'Not found'}`")
                        st.markdown(f"**📱 Phone:** `{data.phone or 'Not found'}`")
                        st.markdown(f"**💼 Experience:** `{data.years_of_experience}` years")
                    
                    with c2:
                        st.markdown("**🎓 Education:**")
                        if data.education:
                            for edu in data.education:
                                st.markdown(f"- {edu}")
                        else:
                            st.markdown("- Not found")

                    st.markdown("**🛠️ Skills:**")
                    st.markdown(", ".join(data.skills))
                    
                    st.markdown("**🏢 Last 3 Job Titles:**")
                    if data.last_3_job_titles:
                        for title in data.last_3_job_titles:
                            st.markdown(f"- {title}")
                    else:
                        st.markdown("- Not found")
                        
                    with st.expander("📦 View Raw JSON Output"):
                        st.json(data.model_dump())
                        
                except ValidationError as ve:
                    st.error("❌ Failed to validate the LLM's output against the schema.")
                    st.exception(ve)
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.info("💡 **Tip:** Ensure Ollama is running in the background (`ollama serve`) and the specified model is downloaded.")