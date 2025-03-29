import google.generativeai as genai
from googlesearch import search
from .info import API_KEY


genai.configure(api_key=API_KEY)


# converts company name to ticker
def get_ticker(company_name):
    prompt = f"What is the NASDAQ or NYSE stock ticker for the company '{company_name}'? Return ONLY the ticker name as a string and nothing else, or else the repsonse will be rejected."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


# searches google given a query
def google_search(query, num_results=3):
    return list(search(query, num_results=num_results))



