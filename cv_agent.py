import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import inspect
import time

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Define input model for minimal CV generation
class MinimalCVInput(BaseModel):
    name: str = Field(..., description="Full name of the person")
    age: Optional[int] = Field(None, description="Age of the person")
    email: Optional[str] = Field(None, description="Email address")
    specialization: str = Field(..., description="Area of specialization (e.g., DevOps, Frontend, Data Science)")
    experience_years: Optional[int] = Field(None, description="Years of experience in the field")
    education_level: Optional[str] = Field(None, description="Highest level of education (e.g., Bachelor's, Master's)")
    key_skills: Optional[List[str]] = Field(default_factory=list, description="List of key skills if known")

# Define output model for CV
class CVOutput(BaseModel):
    resume_text: str = Field(..., description="Full text of the resume")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improving the resume")
    brainstormed_data: Dict[str, Any] = Field(default_factory=dict, description="All brainstormed data used to create the resume")

def generate_resume_filename_and_content(resume_text: str, name: str) -> tuple[str, str]:
    """
    Generate a filename for the resume and return it along with the content.
    Does not save the file locally.
    """
    try:
        print(f"Generating filename for {name}...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename_name = "".join(c if c.isalnum() or c in (' ', '_') else '' for c in name).rstrip()
        filename = f"resume_{safe_filename_name.replace(' ', '_').lower()}_{timestamp}.txt"
        print(f"✅ Generated filename: {filename}")
        return filename, resume_text
        
    except Exception as e:
        print(f"❌ Error generating filename: {str(e)}")
        # Return a default filename and error message as content in case of failure
        return "error_resume.txt", f"Error generating filename: {str(e)}"

# Counter to track how many times generate_cv is called
call_counter = 0

def generate_cv(name: str, specialization: str, persona: str, **kwargs) -> Dict[str, Any]:
    """
    Generate a professional CV using OpenAI Chat API
    """
    global call_counter
    call_counter += 1
    
    # Log detailed debug information about the call
    print(f"\n{'='*80}")
    print(f"CALL #{call_counter} to generate_cv at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Name: {name}, Specialization: {specialization}, Persona: {persona}")
    
    # Print stack trace to see what's calling this function
    stack = inspect.stack()
    if len(stack) > 1:
        caller = stack[1]
        print(f"Called from: {caller.filename}, line {caller.lineno}, function: {caller.function}")
    print(f"{'='*80}\n")
    
    # Add delay to prevent rapid successive calls
    time.sleep(1)
    
    # Build the prompt
    system_prompt = f"""
    Ты должен писать резюме на русском языке.
    You are an elite professional resume writer and career expert specializing in creating 
    elegant, sophisticated, and highly detailed resumes. Create a polished, 
    comprehensive resume based on minimal input provided AND tailored to this specific persona:
    
    PERSONA: {persona}
    
    Format your response as a valid JSON with these fields:
    {{
      "resume_text": "The complete resume text with proper formatting",
      "suggestions": ["suggestion1", "suggestion2", "suggestion3", "suggestion4", "suggestion5"],
      "brainstormed_data": {{
        "professional_summary": "...",
        "contact_info": {{...}},
        "education": [...],
        "experience": [...],
        "skills": {{...}},
        "certifications": [...],
        "languages": [...]
      }}
    }}
    
    The resume should include:
    - Header with name, contact information
    - Professional Summary (5-6 lines)
    - Work Experience (most detailed section, with 4-6 bullet points per role)
    - Education
    - Skills (categorized if possible)
    - Certifications
    - Languages
    
    Make all section headers in ALL CAPS and include detailed bullet points with accomplishments and metrics.
    """
    print(specialization)
    # Format user message
    user_message = f"Create a professional resume for {name}, a {persona} specialist."
    
    # Add additional information if available
    if "age" in kwargs and kwargs["age"]:
        user_message += f"\nAge: {kwargs['age']}"
    
    if "experience_years" in kwargs and kwargs["experience_years"]:
        user_message += f"\nTotal experience: {kwargs['experience_years']} years in the field"
    
    if "education_level" in kwargs and kwargs["education_level"]:
        user_message += f"\nEducation level: {kwargs['education_level']}"
    
    if "key_skills" in kwargs and kwargs["key_skills"]:
        skills_str = ", ".join(kwargs["key_skills"])
        user_message += f"\nKey skills: {skills_str}"
    
    if "job_titles" in kwargs and kwargs["job_titles"]:
        titles_str = ", ".join(kwargs["job_titles"])
        user_message += f"\nPrevious job titles: {titles_str}"
    
    if "achievements" in kwargs and kwargs["achievements"]:
        achievements_str = "; ".join(kwargs["achievements"])
        user_message += f"\nNotable achievements: {achievements_str}"
        
    if "languages" in kwargs and kwargs["languages"]:
        languages_str = ", ".join(kwargs["languages"])
        user_message += f"\nLanguages: {languages_str}"
    
    user_message += "\nPlease create a complete professional resume with realistic but fictional career history."
    
    # Make API call
    print("Sending request to OpenAI...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content
        
        try:
            result = json.loads(content)
            print("Successfully generated resume content")
            
            resume_content = result.get("resume_text", "")
            filename = "resume_error.txt" # Default filename in case generation fails or text is empty

            # Generate filename and get content if resume_text exists
            if resume_content:
                 filename, resume_content = generate_resume_filename_and_content(resume_content, name)
                 print(f"Prepared resume content and filename: {filename}")
                 
            # Return a dictionary including filename and content
            return {
                "filename": filename,
                "resume_content": resume_content, 
                "suggestions": result.get("suggestions", []),
                "brainstormed_data": result.get("brainstormed_data", {})
            }
            
        except json.JSONDecodeError:
            print("Warning: Response wasn't valid JSON")
            # Return the raw content if JSON parsing fails, with a default filename
            return {
                "filename": "invalid_json_resume.txt",
                "resume_content": content, 
                "suggestions": [],
                "brainstormed_data": {}
            }
            
    except Exception as e:
        print(f"Error generating resume: {str(e)}")
        # Return error details in the dictionary
        return {
            "filename": "error_resume.txt",
            "resume_content": f"Failed to generate resume: {str(e)}",
            "suggestions": [],
            "brainstormed_data": {}
        }