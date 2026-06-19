from google import genai
import os

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

genai = os.getenv("GEMINI_API_KEY")

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)


def generate_remarks(
    name,
    glucose,
    haemoglobin,
    cholesterol,
    risks
):

    prompt = f"""
    Patient Name: {name}

    Glucose: {glucose}
    Haemoglobin: {haemoglobin}
    Cholesterol: {cholesterol}

    Predicted Risks:
    {', '.join(risks)}

    Generate a concise professional health summary.

    Rules:

    - Maximum 4 lines
    - Mention detected risks
    - Recommend medical consultation if necessary
    - Do not diagnose disease
    """

    response = model.generate_content(
        prompt
    )

    return response.text