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


# Verifies dates are in valid format (and reformats if not)
def verify_date(start_date, end_date):
    prompt = f"""
    Given a starting date and an ending date, re-write each to the format "yyyy-mm-dd". 
    Return your answer strictly in the following JSON format (with double quotes around keys and values):

    {{
    "start": "re-formatted start date",
    "end": "re-formatted end date"
    }}

    It is imperative that you follow this format strictly and only return the JSON object — no explanations or extra text. 
    Here are the dates to reformat:
    start date: {start_date}, end date: {end_date}.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    json_data = json.loads(response.text.strip())
    return json_data["start"], json_data["end"]

def downlaod_data(stock, start_date, end_date):
    start, end = verify_date(start_date, end_date)
    ticker = get_ticker(stock)
    data = yf.download(ticker, start=start, end=end)
    data.to_csv(f"{ticker}_stock_data.csv")
    return f"{ticker}_stock_data.csv"