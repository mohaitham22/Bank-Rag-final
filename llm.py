import os
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

genai.configure(api_key=os.getenv("GENAI_API_KEY"))

gemini_model = genai.GenerativeModel("gemini-2.5-flash")


def load_prompt():
    """
    Loads the prompt template from a text file.

    Returns:
        str: The content of the prompt template.
    """
    with open("prompt.txt", "r", encoding="utf-8") as file:
        return file.read()


PROMPT_TEMPLATE = load_prompt()


def ask_gemini(question, context):
    """
    Sends a prompt to the Gemini model and retrieves the response.

    Args:
        question (str): The user's question about a bank.
        context (str): The retrieved chunk text to ground the answer in.

    Returns:
        str: Gemini's grounded answer.
    """
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = gemini_model.generate_content(prompt)
    return response.text
