from bs4 import BeautifulSoup
import requests
import openai
import streamlit as st

def analyze_url(url: str):
    # Step 1: Get the HTML
    url = input("Enter the URL to scrape: ").strip()
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    clean_text = soup.get_text(separator="\n", strip=True)
    #Get API and it's response
    openai.api_key = st.secrets["openai"]["api_key"]
    
    system_message = {
        "role": "system",
        "content": "You are top tier data analyst. Your goal is to extract only meaningful business-relevant information and ignore any unrelated UI content, legal notices, navigation text, or generic phrases."
    }
    
    user_message = {
        "role": "user",
        "content": f"""Analyze this text:\n\n{clean_text}\n\nStep 1: Summarize the company in fluent, neutral business English. Include:
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
    }
    
    # Send to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" if using that
        messages=[system_message, user_message],
        temperature=0.4
    )
    return response['choices'][0]['message']['content']
