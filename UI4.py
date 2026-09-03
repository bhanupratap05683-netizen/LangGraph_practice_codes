import streamlit as st
import json
import re
from typing import List
import ollama
from pydantic import BaseModel, Field

# ----------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Resume Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# 2. Non-Nullable Schema (Forces the model to populate)
# ----------------------------------------------------
class ResumeData(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Candidate's email address (use empty string '' if not found)")
    phone: str = Field(description="Candidate's phone number with country/area code (use empty string '' if not found)")
    years_of_experience: float = Field(description="Total calculated work experience in years (e.g. 4.0)")
    skills: List[str] = Field(description="All technical skills, programming languages, and analytical tools mentioned")
    education: List[str] = Field(description="List of degrees, universities, or schools attended")
    last_3_job_titles: List[str] = Field(description="List of job titles held by candidate")


# ----------------------------------------------------
# 3. Regex Fallback Helpers
# ----------------------------------------------------
def extract_email_fallback(text: str) -> str:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else ""

def extract_phone_fallback(text: str) -> str:
    match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,5}", text)
    return match.group(0).strip() if match else ""


# ----------------------------------------------------
# 4. PDF Text Extraction (From Bytes)
# ----------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts text from uploaded PDF bytes using PyMuPDF."""
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF is required. Install via 'pip install pymupdf'")
    
    text = ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text


# ----------------------------------------------------
# 5. Main Extraction Function
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
            {"role": "user", "content": f"Extract all details from this resume text:\n\n{raw_text}"},
        ],
        format=ResumeData.model_json_schema(),
        options={
            "temperature": 0.0,
            "num_ctx": 4096,
        },
    )

    result = ResumeData.model_validate_json(response.message.content)

    # Fallback checks
    if not result.email:
        result.email = extract_email_fallback(raw_text)
    if not result.phone:
        result.phone = extract_phone_fallback(raw_text)

    # Ensure max 3 titles cleanly
    result.last_3_job_titles = result.last_3_job_titles[:3]

    return result


# ----------------------------------------------------
# 6. Streamlit UI Layout
# ----------------------------------------------------
def main():
    st.title("📄 AI Resume Parser")
    st.markdown("Upload a PDF resume to extract structured candidate data using local LLMs (Ollama).")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        model_name = st.text_input("Ollama Model Name", value="qwen2.5-coder:7b")
        st.markdown("---")
        st.info("**How it works:**\n1. Upload a PDF resume.\n2. Text is extracted locally via PyMuPDF.\n3. Ollama parses it into structured JSON.\n4. Regex fallbacks ensure email/phone are never missed.")
        
        # Dependency check
        if not HAS_PYMUPDF:
            st.error("⚠️ PyMuPDF not installed. Run: `pip install pymupdf`")
        
        # Quick Ollama health check
        try:
            ollama.list()
            st.success("✅ Ollama is running")
        except Exception:
            st.error("❌ Ollama is not running. Start it and try again.")

    # Main area: File Uploader
    uploaded_file = st.file_uploader("Upload a Resume (PDF)", type=["pdf"])

    if uploaded_file is not None:
        st.divider()
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            if st.button("🚀 Parse Resume", type="primary", use_container_width=True):
                with st.status("Processing resume...", expanded=True) as status:
                    try:
                        # Step 1: Extract Text
                        status.write("📖 Extracting text from PDF...")
                        raw_text = extract_text_from_pdf_bytes(uploaded_file.read())
                        
                        if not raw_text.strip():
                            st.warning("⚠️ No text could be extracted. The PDF might be image-only (scanned).")
                            status.update(label="Failed: No text found", state="error")
                            st.stop()
                        
                        status.write("🧠 Sending to Ollama for AI parsing...")
                        
                        # Step 2: AI Parsing
                        parsed_data = parse_resume_local(raw_text, model_name=model_name)
                        status.update(label="Parsing complete!", state="complete")

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.stop()

                # Step 3: Display Results
                st.success("✅ Resume parsed successfully!")
                
                # Create tabs for organized viewing
                tab1, tab2 = st.tabs(["📊 Structured JSON", "📝 Extracted Raw Text"])
                
                with tab1:
                    st.markdown("### Parsed Candidate Data")
                    # Display as formatted JSON (Streamlit adds a nice copy button automatically)
                    st.json(parsed_data.model_dump())
                    
                    # Optional: Display as a clean markdown table for quick reading
                    st.markdown("### Quick Summary")
                    summary_col1, summary_col2 = st.columns(2)
                    with summary_col1:
                        st.markdown(f"**Name:** {parsed_data.name}")
                        st.markdown(f"**Email:** {parsed_data.email}")
                        st.markdown(f"**Phone:** {parsed_data.phone}")
                        st.markdown(f"**Experience:** {parsed_data.years_of_experience} years")
                    with summary_col2:
                        st.markdown(f"**Top Skills:** {', '.join(parsed_data.skills[:5])}{'...' if len(parsed_data.skills) > 5 else ''}")
                        st.markdown(f"**Recent Roles:** {', '.join(parsed_data.last_3_job_titles)}")

                with tab2:
                    st.markdown("### Raw Extracted Text (for debugging)")
                    st.text_area("Raw Text", value=raw_text, height=400, disabled=True)

if __name__ == "__main__":
    main()