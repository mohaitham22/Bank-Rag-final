import os
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

genai.configure(api_key=os.getenv("GENAI_API_KEY"))

gemini_model = genai.GenerativeModel("gemini-2.5-flash")


def load_prompt():

    with open("prompt.txt", "r", encoding="utf-8") as file:
        return file.read()


PROMPT_TEMPLATE = load_prompt()


def ask_gemini(question, context):

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = gemini_model.generate_content(prompt)
    return response.text
