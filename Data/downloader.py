import yfinance as yf
from .info import API_KEY
import google.generativeai as genai
import json

genai.configure(api_key=API_KEY)


# Verifies that the ticker is correct (and gets the correct ticker symbol if not)
def get_ticker(company_name):
    prompt = f"What is the NASDAQ or NYSE stock ticker for the company known by '{company_name}'? Return ONLY the ticker name as a string and nothing else, or else the repsonse will be rejected."
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def downlaod_data(stock, start_date, end_date):
    ticker = get_ticker(stock)
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f"{ticker}_stock_data.csv")
    return f"{ticker}_stock_data.csv"