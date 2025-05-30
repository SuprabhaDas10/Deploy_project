from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Google API key from environment
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Create the Gemini model with explicit API key argument
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=google_api_key
)

def extract_news(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # raise error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Failed to fetch news from {url}: {e}"

# Prompt template for summarization
summarize_prompt = PromptTemplate(
    template="Summarize the following news article:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)

# Create the LLMChain
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

def summarize_news(url):
    print(f"\nFetching news from: {url}")
    article = extract_news(url)
    if article.startswith("Failed to fetch"):
        return article
    summary = summarize_chain.run(article=article)
    return summary

if __name__ == "__main__":
    # Take URL input from user
    user_url = input("Please enter your URL:\n").strip()
    result = summarize_news(user_url)
    print(f"\n🔗 {user_url}\n📝 Summary:\n{result}")
