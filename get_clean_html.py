import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI
from functools import lru_cache

def compose_message(clean_text: str) -> str:
    return f"""Analyze this text:\n\n{clean_text}\n\nStep 1: Summarize the company in fluent, neutral business English. Include:
        - Primary industry and sub-industry
        - Business model (e.g., B2B wholesale, D2C retail, SaaS licensing)
        - Core products or services
        - Customer types (e.g., industrial clients, ecommerce, distributors)
        - Geographic presence if available

        Step 2: Immediately after the summary, return 20 to 40 precise business keyword phrases that best describe the company’s offerings.
        - Capitalized like a proper noun (e.g., 'LED Lighting Distribution')
        - Separated by ' OR '
        - Relevant to operations, services, technologies, and value proposition
        - Free of vague terms like 'About Us', 'Contact', 'Team', 'Main Office', etc.

        Respond in this format:

        ---
        SUMMARY:
        <short paragraph>

        KEYWORDS:
        Keyword Phrase 1 OR Keyword Phrase 2 OR Keyword Phrase 3 OR ...
        ---"""

@lru_cache(maxsize=32)
def fetch_and_clean_url(url: str) -> str:
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove UI-related elements
    for tag in soup(['script', 'style', 'nav', 'footer']):
        tag.decompose()

    text = soup.get_text(separator='\n', strip=True)
    return text[:10000]
    
def analyze_url(url: str):
    client_secret = st.secrets["openai"]["api_key"]
    client = OpenAI(api_key=client_secret)

    system_input = "You are top tier data analyst. Your goal is to extract only meaningful business-relevant information and ignore any unrelated UI content, legal notices, navigation text, or generic phrases."
    client_input = compose_message(clean_text)

    # Send to OpenAI
    answerme = client.responses.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" if using that
        instructions = system_input,
        input= client_input,
        temperature=0.4
    )

    return answerme.output_text
    <short paragraph>
    
    KEYWORDS:
    Keyword Phrase 1 OR Keyword Phrase 2 OR Keyword Phrase 3 OR ...
    ---"""
    }
    
    # Send to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" if using that
        messages=[system_message, user_message],
        temperature=0.4
    )
    return response['choices'][0]['message']['content']
