import os
import time
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import PyPDF2
from flask_cors import CORS

# ==============================
# CONFIG
# ==============================
app = Flask(__name__)
CORS(app)

# Configure Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# GEMINI SETUP
# ==============================
# Use gemini-2.5-flash-preview-09-2025 as it is the supported model in this environment
API_KEY = os.environ.get("GEMINI_API_KEY", "") 
MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ==============================
# HELPER FUNCTIONS
# ==============================
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def query_gemini_with_retry(prompt, system_instruction=None):
    """
    Queries Gemini with exponential backoff: 
    Retries up to 5 times with delays of 1s, 2s, 4s, 8s, 16s.
    """
    retries = 5
    for i in range(retries):
        try:
            # We use the system_instruction if provided for better role-playing
            if system_instruction:
                chat_model = genai.GenerativeModel(
                    model_name=MODEL_NAME,
                    system_instruction=system_instruction
                )
                response = chat_model.generate_content(prompt)
            else:
                response = model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            if i == retries - 1:
                return f"AI Error after 5 retries: {str(e)}"
            # Exponential backoff: 2^i seconds
            time.sleep(2**i)
    return "Unknown AI Error"

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    """Serves the frontend interface."""
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Handles the resume analysis and matching."""
    
    # 1. Validate Inputs
    if "resume" not in request.files:
        return jsonify({"error": "Resume PDF is required"}), 400
    
    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")

    if not resume_file.filename:
        return jsonify({"error": "No selected file"}), 400
    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    try:
        # 2. Save and Read PDF
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
        resume_file.save(pdf_path)
        resume_text = extract_text_from_pdf(pdf_path)
        
        # 3. System Instructions
        recruiter_sys = "You are an expert technical recruiter. Analyze the text provided and format the output in clean Markdown bullet points."
        ats_sys = "You are an advanced Applicant Tracking System (ATS). Compare the resume against the job description and provide a match score."

        # 4. Generate Initial Analysis
        resume_prompt = f"Analyze this resume and list technical skills, experience summary, and education:\n\n{resume_text}"
        jd_prompt = f"Analyze this Job Description and list required skills, responsibilities, and preferred qualifications:\n\n{jd_text}"

        parsed_resume = query_gemini_with_retry(resume_prompt, recruiter_sys)
        parsed_jd = query_gemini_with_retry(jd_prompt, recruiter_sys)

        # 5. ATS Matching Prompt
        ats_prompt = f"""
        Compare the parsed resume against the job description.
        
        Resume Analysis:
        {parsed_resume}
        
        Job Description Analysis:
        {parsed_jd}
        
        Output Requirements:
        1. Start exactly with "Match percentage: XX%"
        2. List Matching Skills.
        3. List Missing Keywords/Skills.
        4. Provide 3 tips for improvement.
        """
        
        ats_result = query_gemini_with_retry(ats_prompt, ats_sys)

        return jsonify({
            "parsed_resume": parsed_resume,
            "parsed_job_description": parsed_jd,
            "ats_result": ats_result
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    # Ensure you set GEMINI_API_KEY in your environment variables

    app.run(debug=True, port=8080)
