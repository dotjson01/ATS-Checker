import streamlit as st
import streamlit.components.v1 as components
import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader

# Load environment variables and configure API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to get Gemini output
def get_gemini_output(pdf_text, prompt):
    response = model.generate_content([pdf_text, prompt])
    return response.text

# Function to get ATS score from analysis
def extract_ats_score(analysis_text):
    try:
        # Look for patterns like "ATS SCORE: 75.5" or "ATS Score: 75.5/100"
        import re

        # First, look for the exact ATS SCORE line in the analysis
        # This is the most reliable way to get the exact score as shown in the analysis
        score_line_pattern = r"(?:^|\n)(?:.*?)ATS\s+SCORE:?\s*(.*?)(?:\n|$)"
        score_line_match = re.search(score_line_pattern, analysis_text, re.IGNORECASE | re.MULTILINE)

        if score_line_match:
            # Extract the full score line as displayed in the analysis
            score_line = score_line_match.group(1).strip()

            # Try to extract just the number from this line
            number_match = re.search(r"(\d+\.?\d*)", score_line)
            if number_match:
                score_value = float(number_match.group(1))
                print(f"Found exact score: {score_value} from line: {score_line}")
                # Store both the numeric value and the full text representation
                return {
                    "value": score_value,
                    "display": score_line  # This preserves the exact format shown in the analysis
                }

        # If we couldn't find a specific ATS SCORE line, try more generic patterns
        patterns = [
            r"ATS\s+SCORE:?\s*(\d+\.?\d*)",  # ATS SCORE: 75.5
            r"ATS\s+SCORE:?\s*(\d+\.?\d*)\/100",  # ATS SCORE: 75.5/100
            r"SCORE:?\s*(\d+\.?\d*)",  # SCORE: 75.5
            r"(\d+\.?\d*)/100",  # 75.5/100
            r"^(\d+\.?\d*)$"  # Just a number like 75.5
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE | re.MULTILINE)
            if match:
                score_value = float(match.group(1))
                print(f"Found score: {score_value} using pattern: {pattern}")
                return {
                    "value": score_value,
                    "display": f"{score_value}/100"  # Default display format
                }

        # If no pattern matches, try to find any number in the text
        numbers = re.findall(r"(\d+\.?\d*)", analysis_text)
        if numbers:
            for num in numbers:
                try:
                    score_value = float(num)
                    if 0 <= score_value <= 100:  # Ensure it's a valid score
                        print(f"Found score from numbers: {score_value}")
                        return {
                            "value": score_value,
                            "display": f"{score_value}/100"  # Default display format
                        }
                except Exception:
                    continue

        print("No score found in text:", analysis_text[:100])  # Print first 100 chars for debugging
        return {
            "value": 0,
            "display": "0/100"  # Default when no score is found
        }
    except Exception as e:
        print(f"Error extracting score: {str(e)}")
        return {
            "value": 0,
            "display": "0/100"  # Default on error
        }

# Function to analyze edited resume and return new score
def analyze_edited_resume(edited_text, job_description, ats_model="Generic ATS", job_level="", job_role=""):
    # Get selected ATS system information
    selected_ats = ATS_SYSTEMS[ats_model]

    prompt = f"""
    You are ResumeChecker, an expert in ATS (Applicant Tracking System) analysis.
    Your task is to provide a consistent and accurate evaluation of the resume against the job description,
    specifically for the {ats_model} ATS system.

    ABOUT THE {ats_model.upper()} ATS SYSTEM:
    {selected_ats["description"]}

    KEY FEATURES:
    {', '.join(selected_ats["key_features"])}

    FORMAT PREFERENCES:
    {selected_ats["format_preferences"]}

    PARSING QUIRKS:
    {selected_ats["parsing_quirks"]}

    ANALYSIS APPROACH:
    1. First, thoroughly analyze the job description to identify:
       - Required skills, qualifications, and experience
       - Essential keywords and phrases the ATS will likely scan for
       - Core responsibilities and expectations
       - Industry-specific terminology and jargon

    2. Then, analyze the resume to determine:
       - How well it matches the job requirements
       - Which critical keywords are present or missing
       - If the format is optimized for {ats_model} ATS parsing
       - Whether experience and qualifications align with the job

    3. Calculate the ATS score based on the following criteria with exact weights:
       - Keyword match (40%): Presence of key skills, technologies, and qualifications from the job description
       - Resume format (20%): Proper structure, section organization, and machine readability specifically for {ats_model}
       - Experience relevance (25%): How well the experience matches the job requirements
       - Education match (15%): Relevance of education to the position

    IMPORTANT: Return ONLY the ATS score as a number out of 100 with one decimal place precision.
    Do not include any other text, explanation, or analysis.

    Resume text: {edited_text}
    Job description: {job_description}
    Job level: {job_level}
    Job role: {job_role}
    """
    response = model.generate_content([edited_text, prompt])
    try:
        score_text = response.text.strip()
        # Extract score using our improved function
        score_result = extract_ats_score(score_text)
        return score_result
    except Exception as e:
        print(f"Error in analyze_edited_resume: {str(e)}")
        return {"value": 0, "display": "0/100"}

# Function to read PDF
def read_pdf(uploaded_file):
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
        return pdf_text
    else:
        raise FileNotFoundError("No file uploaded")

# ATS system information
ATS_SYSTEMS = {
    "Generic ATS": {
        "description": "A standard ATS that uses keyword matching and basic resume parsing.",
        "key_features": [
            "Keyword matching",
            "Basic resume parsing",
            "Standard formatting requirements"
        ],
        "format_preferences": "Standard resume format with clear section headings (Summary, Experience, Skills, Education).",
        "parsing_quirks": "May struggle with complex formatting, tables, and graphics."
    },
    "iCIMS": {
        "description": "A comprehensive talent acquisition platform used by many large enterprises.",
        "key_features": [
            "Advanced keyword matching",
            "Semantic search capabilities",
            "Skills-based filtering"
        ],
        "format_preferences": "Clean formatting with standard section headers. Supports DOC, DOCX, PDF, RTF, and TXT formats.",
        "parsing_quirks": "Better at parsing PDF files than some other systems. May have issues with headers/footers and complex tables."
    },
    "Greenhouse": {
        "description": "A hiring software platform focused on structured hiring processes.",
        "key_features": [
            "Attribute-based candidate evaluation",
            "Custom application questions",
            "Collaborative hiring"
        ],
        "format_preferences": "Clean, simple formatting. Works well with standard chronological resumes.",
        "parsing_quirks": "May miss information in non-standard sections. Handles PDF and Word documents well."
    },
    "Manatal": {
        "description": "An AI-powered recruitment software with advanced candidate matching.",
        "key_features": [
            "AI-powered candidate matching",
            "Social media enrichment",
            "Multilingual support"
        ],
        "format_preferences": "Standard resume format with clear section delineation. Supports various file formats.",
        "parsing_quirks": "AI capabilities help with understanding context, but may still struggle with highly creative formats."
    },
    "ClearCompany": {
        "description": "A talent management platform with emphasis on company goals and culture fit.",
        "key_features": [
            "Goal alignment",
            "Culture-based screening",
            "Competency mapping"
        ],
        "format_preferences": "Traditional resume format with clear sections. Prefers chronological format.",
        "parsing_quirks": "May prioritize experience descriptions that align with company values and goals."
    },
    "Bullhorn": {
        "description": "A recruitment software popular with staffing and recruiting agencies.",
        "key_features": [
            "Candidate tracking",
            "Resume parsing",
            "Job matching"
        ],
        "format_preferences": "Standard resume formats. Handles various file types including PDF and Word.",
        "parsing_quirks": "Strong at parsing contact information and work history, may struggle with skill categorization in non-standard formats."
    },
    "Transformify": {
        "description": "A modern ATS with focus on diversity and inclusion in hiring.",
        "key_features": [
            "Blind recruitment options",
            "Skills-based matching",
            "Global talent pool"
        ],
        "format_preferences": "Clean, standard formatting. Supports skills-based and chronological formats.",
        "parsing_quirks": "May place higher emphasis on skills and qualifications over chronological work history."
    }
}

# Job description templates
JOB_TEMPLATES = {
    "Fresher SDE": """
Job Title: Software Development Engineer (Entry Level)

About the Role:
We are looking for entry-level Software Development Engineers to join our growing team. This is an excellent opportunity for recent graduates to start their career in software development.

Requirements:
- Bachelor's degree in Computer Science, Engineering, or related field
- Knowledge of at least one programming language (Java, Python, C++, etc.)
- Basic understanding of data structures and algorithms
- Familiarity with software development methodologies
- Strong problem-solving skills
- Good communication and teamwork abilities
- Willingness to learn new technologies

Responsibilities:
- Write clean, maintainable code according to specifications
- Participate in code reviews and implement feedback
- Debug and fix issues in existing applications
- Collaborate with team members on project development
- Learn and adapt to new technologies and frameworks
- Assist in testing and quality assurance
- Document code and processes

Nice to Have:
- Experience with web development (HTML, CSS, JavaScript)
- Knowledge of database systems (SQL, NoSQL)
- Familiarity with version control systems (Git)
- Understanding of cloud platforms (AWS, Azure, GCP)
- Previous internship or project experience
""",

    "Intermediate SDE": """
Job Title: Software Development Engineer II

About the Role:
We are seeking a talented Software Development Engineer with 2-4 years of experience to join our engineering team. In this role, you will design, develop, and maintain software applications while collaborating with cross-functional teams.

Requirements:
- Bachelor's or Master's degree in Computer Science, Engineering, or related field
- 2-4 years of professional software development experience
- Strong proficiency in at least one programming language (Java, Python, C++, etc.)
- Experience with web development frameworks and technologies
- Solid understanding of data structures, algorithms, and software design patterns
- Experience with database systems and SQL
- Familiarity with Agile development methodologies
- Strong problem-solving and analytical skills
- Excellent communication and teamwork abilities

Responsibilities:
- Design, develop, and maintain software applications
- Write clean, efficient, and maintainable code
- Participate in all phases of the software development lifecycle
- Collaborate with product managers, designers, and other engineers
- Conduct code reviews and provide constructive feedback
- Debug complex issues across multiple systems
- Implement automated tests to ensure code quality
- Contribute to technical documentation
- Mentor junior developers

Nice to Have:
- Experience with microservices architecture
- Knowledge of cloud platforms (AWS, Azure, GCP)
- Experience with CI/CD pipelines
- Understanding of containerization (Docker, Kubernetes)
- Contributions to open-source projects
""",

    "Senior SDE": """
Job Title: Senior Software Development Engineer

About the Role:
We are looking for an experienced Senior Software Development Engineer to lead technical initiatives, architect solutions, and mentor junior team members. The ideal candidate will have a strong technical background and leadership skills.

Requirements:
- Bachelor's or Master's degree in Computer Science, Engineering, or related field
- 5+ years of professional software development experience
- Expert-level proficiency in multiple programming languages
- Deep understanding of software architecture and design patterns
- Experience with distributed systems and microservices
- Strong knowledge of database design and optimization
- Experience with cloud platforms (AWS, Azure, GCP)
- Proficiency with CI/CD pipelines and DevOps practices
- Experience leading technical projects and mentoring junior developers
- Excellent problem-solving, communication, and leadership skills

Responsibilities:
- Design and implement complex software systems
- Lead technical initiatives and architectural decisions
- Write high-quality, maintainable, and efficient code
- Review code and provide technical guidance to team members
- Collaborate with product managers to define requirements and solutions
- Identify and resolve technical debt and performance issues
- Implement best practices for software development
- Mentor and coach junior engineers
- Contribute to hiring and team growth
- Stay current with industry trends and emerging technologies

Nice to Have:
- Experience with machine learning or AI technologies
- Knowledge of security best practices
- Experience with high-scale, high-availability systems
- Contributions to open-source projects
- Technical publications or conference presentations
""",

    "Data Analyst": """
Job Title: Data Analyst

About the Role:
We are seeking a detail-oriented Data Analyst to help transform raw data into actionable insights. The ideal candidate will have strong analytical skills and the ability to present complex findings in a clear, understandable manner.

Requirements:
- Bachelor's degree in Statistics, Mathematics, Computer Science, Economics, or related field
- 1-3 years of experience in data analysis or related role
- Proficiency in SQL and experience with databases
- Strong skills in Excel and data visualization tools (Tableau, Power BI, etc.)
- Experience with statistical analysis and data mining
- Knowledge of Python or R for data analysis
- Ability to translate complex data into clear insights and recommendations
- Strong problem-solving and critical thinking skills
- Excellent communication and presentation abilities

Responsibilities:
- Collect, clean, and preprocess data from various sources
- Perform statistical analysis to identify patterns and trends
- Create and maintain dashboards and reports
- Collaborate with stakeholders to understand business requirements
- Develop and implement data collection systems and other strategies
- Monitor and analyze key performance indicators
- Present findings and recommendations to non-technical audiences
- Support data-driven decision making across the organization
- Document processes and maintain data dictionaries

Nice to Have:
- Experience with big data technologies (Hadoop, Spark)
- Knowledge of machine learning techniques
- Understanding of data warehousing concepts
- Experience with A/B testing and experimentation
- Familiarity with business intelligence tools
""",

    "MERN Stack Developer": """
Job Title: MERN Stack Developer

About the Role:
We are looking for a skilled MERN Stack Developer to build and maintain web applications using MongoDB, Express.js, React.js, and Node.js. The ideal candidate will have experience with all aspects of the MERN stack and a passion for creating responsive, user-friendly applications.

Requirements:
- Bachelor's degree in Computer Science, Web Development, or related field
- 2+ years of experience with the MERN stack (MongoDB, Express.js, React.js, Node.js)
- Strong proficiency in JavaScript, including ES6+ features
- Experience with React.js and its core principles (components, props, state, hooks)
- Knowledge of Node.js and Express.js for backend development
- Experience with MongoDB and Mongoose ODM
- Familiarity with RESTful APIs and GraphQL
- Understanding of frontend build tools (Webpack, Babel, etc.)
- Experience with version control systems (Git)
- Knowledge of responsive design and CSS frameworks (Bootstrap, Material-UI, etc.)

Responsibilities:
- Develop and maintain web applications using the MERN stack
- Build reusable components and frontend libraries
- Design and implement RESTful APIs
- Optimize applications for maximum speed and scalability
- Implement security and data protection measures
- Collaborate with designers and other developers
- Debug issues and implement fixes
- Write clean, maintainable, and efficient code
- Stay up-to-date with emerging trends and technologies

Nice to Have:
- Experience with TypeScript
- Knowledge of Redux for state management
- Familiarity with testing frameworks (Jest, Mocha, Cypress)
- Experience with CI/CD pipelines
- Understanding of containerization (Docker)
- Knowledge of AWS or other cloud platforms
- Experience with server-side rendering (Next.js)
"""
}

# Streamlit UI
st.set_page_config(page_title="ATS-Checker - Powered by Gemini AI", layout="wide")

# Custom CSS for a ResumeWorded-like design with hidden header
st.markdown("""
<style>
    /* Hide Streamlit header (Deploy button and menu) */
    header {
        display: none !important;
    }

    /* Hide Streamlit footer */
    footer {
        display: none !important;
    }

    /* Adjust padding to account for removed header */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 100%;
        margin: 0 auto;
    }
    h1 {
        color: #2e4057;
    }
    h2 {
        color: #4d648d;
    }
    .stButton>button {
        background-color: #4d648d;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2e4057;
    }
    .results {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #333333 !important;
        font-size: 16px;
        line-height: 1.6;
    }
    /* Score styles removed */
    .score-improvement {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
        text-align: center;
    }
    .issue-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .issue-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    .issue-icon {
        color: #ff7043;
        margin-right: 10px;
        font-size: 24px;
    }
    .issue-title {
        font-size: 18px;
        font-weight: 600;
        color: #333;
    }
    .issue-count {
        background-color: #f0f0f0;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 14px;
        margin-left: 10px;
    }
    .issue-content {
        color: #555;
        line-height: 1.5;
    }
    .highlight-text {
        background-color: #fff9c4;
        padding: 2px 4px;
    }
    .suggestion-button {
        background-color: #e3f2fd;
        color: #1976d2;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
        margin-top: 10px;
    }
    .fixed-button {
        background-color: #f5f5f5;
        color: #333;
        border: 1px solid #ddd;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
        float: right;
    }
    .resume-editor {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        background-color: white;
    }
    .nav-header {
        background-color: #1a237e;
        color: white;
        padding: 15px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    .nav-title {
        font-weight: bold;
        font-size: 18px;
        letter-spacing: 1px;
    }
    .nav-subtitle {
        opacity: 0.8;
    }
    textarea {
        font-family: monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4d648d;
        color: white;
    }
    .stProgress > div > div {
        background-color: #f8a978;
    }
</style>
""", unsafe_allow_html=True)

# Header - ResumeWorded style
st.markdown("""
<div class="nav-header">
    <div class="nav-title">ATS CHECKER</div>
    <div class="nav-subtitle">POWERED BY GEMINI AI</div>
</div>
""", unsafe_allow_html=True)

# Main content - ResumeWorded style layout with side-by-side view
st.markdown("### Upload your resume and select a job description to get started")

# Create a 3-column layout: left sidebar for inputs and score, middle for analysis, right for editable resume
# Adjusted proportions for better use of screen space
left_col, middle_col, right_col = st.columns([1, 1.5, 1.5])

with left_col:
    # File upload
    upload_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    # Job template selection
    st.subheader("Job Description")
    use_template = st.checkbox("Use a job description template", value=False)

    if use_template:
        job_template = st.selectbox(
            "Select a job template:",
            list(JOB_TEMPLATES.keys())
        )
        st.info(f"Selected template: {job_template}")
        job_description = JOB_TEMPLATES[job_template]
        with st.expander("View Job Description"):
            st.write(job_description)
    else:
        # Enhanced custom job description input
        st.markdown("""
        <div style="margin-bottom: 5px;">
            <span style="font-weight: bold;">Enter Job Description</span>
            <span style="font-size: 0.85em; color: #666; margin-left: 5px;">
                (Paste the complete job posting for best results)
            </span>
        </div>
        """, unsafe_allow_html=True)

        job_description = st.text_area("",
                                      placeholder="Paste the full job description here. Include all requirements, qualifications, and responsibilities for the most accurate analysis.",
                                      height=200)

        if job_description:
            # Show character count and quality indicator
            char_count = len(job_description)
            if char_count < 200:
                st.warning(f"Job description is too short ({char_count} characters). For best results, paste the complete job posting.")
            elif char_count < 500:
                st.info(f"Job description length: {char_count} characters. More details will improve analysis accuracy.")
            else:
                st.success(f"Job description length: {char_count} characters. Good level of detail for accurate analysis.")

    # Job level selection
    job_level = st.selectbox("Select Job Level", ["Entry Level/Fresher", "Intermediate (2-5 years)", "Advanced (5+ years)"])

    # Job role selection
    job_role = st.selectbox("Select Job Role", ["Software Development Engineer", "Data Analyst/Scientist", "MERN Stack Developer", "Other"])

    # ATS system focus selection (with clarification)
    st.markdown("""
    <div style="margin-bottom: 5px;">
        <span style="font-weight: bold;">ATS System Focus</span>
        <span style="font-size: 0.85em; color: #666; margin-left: 5px;">
            (Analysis uses Gemini AI with specialized prompts for each ATS system)
        </span>
    </div>
    """, unsafe_allow_html=True)
    ats_model = st.selectbox("", ["Generic ATS", "iCIMS", "Greenhouse", "Manatal", "ClearCompany", "Bullhorn", "Transformify"],
                            help="Select which ATS system to focus on. The analysis will be tailored with specific knowledge about this system's preferences and behaviors.")

    # Analyze button
    if st.button("Analyze Resume"):
        if upload_file is not None:
            with st.spinner("Analyzing your resume..."):
                try:
                    pdf_text = read_pdf(upload_file)

                    # Get selected ATS system information
                    selected_ats = ATS_SYSTEMS[ats_model]

                    # Store selected ATS in session state
                    st.session_state.selected_ats = ats_model

                    # Use a tailored prompt for ATS analysis with specific ATS information and deep job description analysis
                    prompt = f"""
                    You are ResumeChecker, an expert in ATS (Applicant Tracking System) analysis. Your task is to provide a comprehensive evaluation of the resume against the job description, specifically for the {ats_model} ATS system, and help the candidate pass the ATS screening process.

                    ABOUT THE {ats_model.upper()} ATS SYSTEM:
                    {selected_ats["description"]}

                    KEY FEATURES:
                    {', '.join(selected_ats["key_features"])}

                    FORMAT PREFERENCES:
                    {selected_ats["format_preferences"]}

                    PARSING QUIRKS:
                    {selected_ats["parsing_quirks"]}

                    ANALYSIS APPROACH:
                    1. First, thoroughly analyze the job description to identify:
                       - Required skills, qualifications, and experience
                       - Essential keywords and phrases the ATS will likely scan for
                       - Core responsibilities and expectations
                       - Company values and culture indicators
                       - Industry-specific terminology and jargon

                    2. Then, analyze the resume to determine:
                       - How well it matches the job requirements
                       - Which critical keywords are present or missing
                       - If the format is optimized for {ats_model} ATS parsing
                       - Whether experience and qualifications align with the job

                    3. Calculate the ATS score based on the following criteria with exact weights:
                       - Keyword match (40%): Presence of key skills, technologies, and qualifications from the job description
                       - Resume format (20%): Proper structure, section organization, and machine readability specifically for {ats_model}
                       - Experience relevance (25%): How well the experience matches the job requirements
                       - Education match (15%): Relevance of education to the position

                    ANALYSIS FORMAT:
                    1. JOB DESCRIPTION ANALYSIS:
                       - Summarize the key requirements and qualifications from the job description
                       - List the most important keywords and phrases the ATS will scan for
                       - Identify any unique or specific requirements that stand out

                    2. ATS SCORE: Provide a single, consistent score out of 100 with one decimal place precision

                    3. KEY FINDINGS:
                       - Identify the most important keywords found and missing in the resume
                       - Evaluate the resume structure and format for {ats_model} ATS compatibility
                       - Assess the overall match between the resume and job description

                    4. {ats_model.upper()} SPECIFIC RECOMMENDATIONS:
                       - Provide specific advice for optimizing this resume for the {ats_model} ATS system
                       - Highlight any particular strengths or weaknesses for this specific ATS
                       - Explain how this specific ATS might evaluate certain aspects of the resume

                    5. OPTIMIZATION SUGGESTIONS:
                       - List 5 specific, actionable recommendations to improve the resume for this job
                       - Suggest exact keywords to add and where to place them
                       - Recommend format changes to improve ATS readability
                       - Provide specific phrasing suggestions that align with the job description

                    6. SECTION-BY-SECTION ANALYSIS:
                       - Analyze each major section of the resume (Summary, Experience, Skills, Education)
                       - Provide specific improvement suggestions for each section
                       - Suggest how to better align each section with the job requirements

                    7. ATS PASSING STRATEGY:
                       - Provide a clear strategy for passing the {ats_model} ATS for this specific job
                       - Highlight the most critical changes needed to improve chances of getting through the ATS
                       - Suggest any industry-specific tactics that might help for this particular role

                    Resume text: {pdf_text}
                    Job description: {job_description}
                    Job level: {job_level}
                    Job role: {job_role}
                    """

                    response = get_gemini_output(pdf_text, prompt)

                    # Extract ATS score from the analysis
                    original_score = extract_ats_score(response)

                    # Store in session state
                    st.session_state.analysis_response = response
                    st.session_state.original_score = original_score
                    st.session_state.current_score = original_score
                    st.session_state.pdf_text = pdf_text
                    st.session_state.job_description = job_description
                    st.session_state.job_level = job_level
                    st.session_state.job_role = job_role

                    # Always update the editable resume when a new file is uploaded
                    st.session_state.edited_resume = pdf_text

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please upload a resume to analyze.")

# Display analysis results if available
if 'analysis_response' in st.session_state:
    # Left column - Score display (ResumeWorded style)
    with left_col:
        st.markdown("## ATS Score")

        # Display selected ATS system focus
        ats_model = st.session_state.selected_ats
        st.markdown(f"""
        <div>
            <span style="font-weight: bold; font-size: 1.1em;">ATS System Focus: {ats_model}</span>
            <span style="display: block; font-size: 0.85em; color: #666; margin-top: 3px;">
                Analysis powered by Google's Gemini AI with specialized knowledge of this ATS system
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Show ATS system description
        with st.expander("About this ATS system"):
            st.write(ATS_SYSTEMS[ats_model]["description"])
            st.write("**Key Features:**")
            for feature in ATS_SYSTEMS[ats_model]["key_features"]:
                st.write(f"- {feature}")
            st.write(f"**Format Preferences:** {ATS_SYSTEMS[ats_model]['format_preferences']}")
            st.write("---")
            st.write("**How this works:** Our AI analyzes your resume using specialized knowledge about this ATS system's preferences and behaviors. While we use Google's Gemini model for all analyses, the prompts and evaluation criteria are tailored specifically for each ATS system based on research and industry knowledge.")

        # Display the exact score as it appears in the analysis
        score = st.session_state.current_score
        st.markdown(f"### Current Score: {score['display']}", unsafe_allow_html=True)

        # Score improvement if changes were made
        if st.session_state.current_score['value'] > st.session_state.original_score['value']:
            improvement = st.session_state.current_score['value'] - st.session_state.original_score['value']
            st.markdown(f"""
            <div class="score-improvement">
                +{improvement:.1f} POINTS
            </div>
            """, unsafe_allow_html=True)

        # Extract issues from analysis
        analysis_text = st.session_state.analysis_response

        # Simplified issue extraction (in a real app, this would be more sophisticated)
        missing_keywords = []
        format_issues = []
        suggestions = []

        # Very basic extraction logic - in a real app this would be more robust
        if "missing keywords" in analysis_text.lower():
            missing_keywords = ["Add relevant keywords from job description"]

        if "format" in analysis_text.lower():
            format_issues = ["Improve resume formatting for ATS"]

        if "suggest" in analysis_text.lower() or "recommendation" in analysis_text.lower():
            suggestions = ["Implement suggested improvements"]

        # Navigation menu for issues (similar to ResumeWorded)
        st.markdown("## ISSUES")

        # Display issue categories
        issues = {
            "Keyword Match": len(missing_keywords),
            "Format Issues": len(format_issues),
            "Content Suggestions": len(suggestions)
        }

        for issue, count in issues.items():
            if count > 0:
                st.markdown(f"### {issue} ({count})")

    # Middle column - Analysis details
    with middle_col:
        st.markdown("## Analysis Results")

        # Display the full analysis with proper formatting and colors
        st.markdown(f'<div class="results" style="width: 100%; overflow-wrap: break-word;">{st.session_state.analysis_response}</div>', unsafe_allow_html=True)

        # Option to chat about the resume
        st.markdown("### Have questions about your resume?")
        user_question = st.text_input("Ask me anything about your resume or the analysis:")

        if user_question:
            with st.spinner("Generating response..."):
                chat_prompt = f"""
                You are ResumeChecker, an expert in ATS (Applicant Tracking System) analysis.
                Based on the resume and previous analysis, answer the following question with specific,
                actionable advice. Be consistent in your responses and maintain the same evaluation criteria
                used in the original analysis.

                Question: {user_question}

                Resume text: {st.session_state.pdf_text}
                Job description: {st.session_state.job_description}
                Previous analysis: {st.session_state.analysis_response}
                """

                chat_response = get_gemini_output(st.session_state.pdf_text, chat_prompt)
                st.markdown(f'<div class="results" style="width: 100%; overflow-wrap: break-word;">{chat_response}</div>', unsafe_allow_html=True)

    # Right column - Editable resume with live updates
    with right_col:
        st.markdown("## Edit Your Resume")
        st.markdown("Make changes to your resume based on the suggestions and see your score improve in real-time.")

        # Editable resume text area with ResumeWorded-like styling - using more space
        st.markdown('<div class="resume-editor" style="width: 100%;">', unsafe_allow_html=True)
        edited_resume = st.text_area("",
                                    value=st.session_state.edited_resume,
                                    height=500,  # Increased height
                                    key="resume_editor")
        st.markdown('</div>', unsafe_allow_html=True)

        # Update the editable resume in session state
        st.session_state.edited_resume = edited_resume

        # Button to analyze the updated resume
        if st.button("Update Score"):
            if edited_resume:
                # Check if the resume has actually been edited
                if edited_resume != st.session_state.pdf_text:
                    with st.spinner("Updating score..."):
                        # Get updated ATS analysis using the selected ATS model and job details
                        new_score = analyze_edited_resume(
                            edited_resume,
                            st.session_state.job_description,
                            st.session_state.selected_ats,
                            job_level=st.session_state.get('job_level', ''),
                            job_role=st.session_state.get('job_role', '')
                        )

                        # Update the score in session state
                        st.session_state.current_score = new_score

                        # Show the exact score from analysis
                        if new_score['value'] > st.session_state.original_score['value']:
                            improvement = new_score['value'] - st.session_state.original_score['value']
                            st.success(f"Your resume received an ATS Score of {new_score['display']} (improved by {improvement:.1f} points)")
                        elif new_score['value'] < st.session_state.original_score['value']:
                            decrease = st.session_state.original_score['value'] - new_score['value']
                            st.error(f"Your resume received an ATS Score of {new_score['display']} (decreased by {decrease:.1f} points)")
                        else:
                            st.info(f"Your resume received an ATS Score of {new_score['display']} (unchanged)")
                else:
                    # No changes made, keep the original score
                    st.info("No changes detected in the resume. Score remains the same.")
            else:
                st.error("Resume text cannot be empty.")

# Get current year and month
current_date = datetime.now().strftime('%Y %B')

# Create a footer container with background
st.markdown("""
<div style="background-color: #f5f5f5; padding: 20px; margin-top: 30px; border-top: 1px solid #ddd; text-align: center; width: 100%;">
""", unsafe_allow_html=True)

# Main footer content
st.markdown("""
<div style="display: flex; justify-content: center; max-width: 1600px; margin: 0 auto 20px auto; flex-wrap: wrap; gap: 40px;">
    <div>
        <h4 style="color: #333; margin-bottom: 10px;">ATS CHECKER</h4>
        <p style="color: #666; font-size: 14px;">Optimize your resume for ATS systems</p>
    </div>
    <div>
        <h4 style="color: #333; margin-bottom: 10px;">RESOURCES</h4>
        <p style="color: #666; font-size: 14px;">
            <a href="https://career.io/career-advice/create-an-optimized-ats-resume" target="_blank" style="color: #4d648d; text-decoration: none;">ATS Guide</a> |
            <a href="https://cdn-careerservices.fas.harvard.edu/wp-content/uploads/sites/161/2023/08/College-resume-and-cover-letter-4.pdf" target="_blank" style="color: #4d648d; text-decoration: none;">Resume Tips</a>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Developer section with footer
st.markdown("---")
st.markdown("<h4 style='text-align: center; color: #333; margin-bottom: 20px;'>DEVELOPED BY</h4>", unsafe_allow_html=True)

# Developer 1
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="display: flex; align-items: center; background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border: 1px solid #eee; margin: 0 auto 20px auto;">
        <img src="https://github.com/thestarsahil.png" alt="Sahil" style="width: 70px; height: 70px; border-radius: 50%; margin-right: 15px; border: 3px solid #4d648d; object-fit: cover;">
        <div>
            <h4 style="margin: 0; color: #333; font-size: 18px;">Sahil Ali</h4>
            <p style="margin: 5px 0 0; color: #666; font-size: 13px;">C++ Embedded Programmer</p>
            <p style="margin: 5px 0 0; color: #777; font-size: 12px;">Computer Science Student</p>
            <a href="https://github.com/thestarsahil" target="_blank" style="color: #4d648d; text-decoration: none; font-size: 12px; display: inline-block; margin-top: 5px;">GitHub Profile</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align: center; color: #999; font-size: 12px; margin-top: 20px;'>Copyright " + current_date + " ATS-Checker | Made by Embedded Programmers Society</div>", unsafe_allow_html=True)
# making 
#light way to use the app