import streamlit as st
import os
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
        score_pattern = r"ATS\s+SCORE:?\s*(\d+\.?\d*)"
        match = re.search(score_pattern, analysis_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0
    except:
        return 0

# Function to analyze edited resume and return new score
def analyze_edited_resume(edited_text, job_description):
    prompt = f"""
    You are ResumeChecker, an expert in ATS (Applicant Tracking System) analysis.
    Your task is to provide a consistent and accurate evaluation of the resume against the job description.

    INSTRUCTIONS:
    1. Use a deterministic scoring algorithm that will produce the same score for the same resume and job description every time.
    2. Calculate the ATS score based on the following criteria with exact weights:
       - Keyword match (40%): Presence of key skills, technologies, and qualifications from the job description
       - Resume format (20%): Proper structure, section organization, and machine readability
       - Experience relevance (25%): How well the experience matches the job requirements
       - Education match (15%): Relevance of education to the position

    IMPORTANT: Return ONLY the ATS score as a number out of 100 with one decimal place precision.
    Do not include any other text, explanation, or analysis.

    Resume text: {edited_text}
    Job description: {job_description}
    """
    response = model.generate_content([edited_text, prompt])
    try:
        score_text = response.text.strip()
        # Try to extract just the number
        import re
        score_match = re.search(r"(\d+\.?\d*)", score_text)
        if score_match:
            return float(score_match.group(1))
        return float(score_text)
    except:
        return 0

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
st.set_page_config(page_title="ATS-Checker", layout="wide")

# Custom CSS for a ResumeWorded-like design
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
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
        color: #333333;
    }
    .score-circle {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: white;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin: 0 auto;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        position: relative;
    }
    .score-circle::before {
        content: '';
        position: absolute;
        top: -10px;
        left: -10px;
        right: -10px;
        bottom: -10px;
        border-radius: 50%;
        border: 10px solid #f8a978;
        border-top-color: #f8a978;
        border-right-color: #f8a978;
        border-bottom-color: #f8a978;
        border-left-color: #f8a978;
        clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
        opacity: 0.7;
    }
    .score-number {
        font-size: 48px;
        font-weight: bold;
        color: #333;
    }
    .score-label {
        font-size: 14px;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
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
    <div class="nav-subtitle">SCORE MY RESUME</div>
</div>
""", unsafe_allow_html=True)

# Main content - ResumeWorded style layout with side-by-side view
st.markdown("### Upload your resume and select a job description to get started")

# Create a 3-column layout: left sidebar for score, middle for analysis, right for editable resume
left_col, middle_col, right_col = st.columns([1, 2, 2])

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
        # Custom job description input
        job_description = st.text_area("Enter the job description (optional)", height=150)

    # Job level selection
    job_level = st.selectbox("Select Job Level", ["Entry Level/Fresher", "Intermediate (2-5 years)", "Advanced (5+ years)"])

    # Job role selection
    job_role = st.selectbox("Select Job Role", ["Software Development Engineer", "Data Analyst/Scientist", "MERN Stack Developer", "Other"])

    # Analyze button
    if st.button("Analyze Resume"):
        if upload_file is not None:
            with st.spinner("Analyzing your resume..."):
                try:
                    pdf_text = read_pdf(upload_file)

                    # Use a single, consistent prompt for ATS analysis
                    prompt = f"""
                    You are ResumeChecker, an expert in ATS (Applicant Tracking System) analysis. Your task is to provide a consistent and accurate evaluation of the resume against the job description.

                    INSTRUCTIONS:
                    1. Use a deterministic scoring algorithm that will produce the same score for the same resume and job description every time.
                    2. Calculate the ATS score based on the following criteria with exact weights:
                       - Keyword match (40%): Presence of key skills, technologies, and qualifications from the job description
                       - Resume format (20%): Proper structure, section organization, and machine readability
                       - Experience relevance (25%): How well the experience matches the job requirements
                       - Education match (15%): Relevance of education to the position

                    ANALYSIS FORMAT:
                    1. ATS SCORE: Provide a single, consistent score out of 100 with one decimal place precision
                    2. KEY FINDINGS:
                       - Identify the most important keywords found and missing in the resume
                       - Evaluate the resume structure and format for ATS compatibility
                       - Assess the overall match between the resume and job description
                    3. OPTIMIZATION SUGGESTIONS:
                       - List 5 specific, actionable recommendations to improve the resume for this job
                       - Suggest exact keywords to add and where to place them
                       - Recommend format changes to improve ATS readability
                    4. SECTION-BY-SECTION ANALYSIS:
                       - Briefly analyze each major section of the resume (Summary, Experience, Skills, Education)
                       - Provide specific improvement suggestions for each section

                    Resume text: {pdf_text}
                    Job description: {job_description}
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

                    # Initialize editable resume if not already present
                    if 'edited_resume' not in st.session_state:
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

        # Score circle with percentage
        score = st.session_state.current_score
        st.markdown(f"""
        <div class="score-circle">
            <div class="score-number">{int(score)}</div>
            <div class="score-label">OVERALL</div>
        </div>
        """, unsafe_allow_html=True)

        # Score improvement if changes were made
        if st.session_state.current_score > st.session_state.original_score:
            improvement = st.session_state.current_score - st.session_state.original_score
            st.markdown(f"""
            <div class="score-improvement">
                +{int(improvement)} POINTS
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
        st.markdown(f'<div class="results">{st.session_state.analysis_response}</div>', unsafe_allow_html=True)

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
                st.markdown(f'<div class="results">{chat_response}</div>', unsafe_allow_html=True)

    # Right column - Editable resume with live updates
    with right_col:
        st.markdown("## Edit Your Resume")
        st.markdown("Make changes to your resume based on the suggestions and see your score improve in real-time.")

        # Editable resume text area with ResumeWorded-like styling
        st.markdown('<div class="resume-editor">', unsafe_allow_html=True)
        edited_resume = st.text_area("",
                                    value=st.session_state.edited_resume,
                                    height=400,
                                    key="resume_editor")
        st.markdown('</div>', unsafe_allow_html=True)

        # Update the editable resume in session state
        st.session_state.edited_resume = edited_resume

        # Button to analyze the updated resume
        if st.button("Update Score"):
            if edited_resume:
                with st.spinner("Updating score..."):
                    # Get updated ATS analysis
                    new_score = analyze_edited_resume(edited_resume, st.session_state.job_description)

                    # Update the score in session state
                    st.session_state.current_score = new_score

                    # Show score improvement
                    if new_score > st.session_state.original_score:
                        improvement = new_score - st.session_state.original_score
                        st.success(f"🎉 Your resume score improved by {improvement:.1f} points!")
                    elif new_score < st.session_state.original_score:
                        decrease = st.session_state.original_score - new_score
                        st.error(f"⚠️ Your resume score decreased by {decrease:.1f} points. Try different changes.")
                    else:
                        st.info("Your score remains the same. Try implementing more suggestions.")
            else:
                st.error("Resume text cannot be empty.")

# Footer with ResumeWorded-like styling
st.markdown("""
<div style="background-color: #f5f5f5; padding: 20px; margin-top: 30px; border-top: 1px solid #ddd; text-align: center;">
    <div style="display: flex; justify-content: space-between; max-width: 800px; margin: 0 auto;">
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
    <p style="color: #999; font-size: 12px; margin-top: 20px;">© 2024 ATS-Checker | Made with ❤️ to help job seekers</p>
</div>
""", unsafe_allow_html=True)
