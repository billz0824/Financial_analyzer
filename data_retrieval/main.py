import os
from .data_tools import get_ticker, google_search
from .sec_scraper import get_clean_10k

def retrieve_company_documents(company_name, base_path="data"):
    ticker = get_ticker(company_name)
    if not ticker:
        raise ValueError(f"Ticker not found for {company_name}")

    company_dir = os.path.join(base_path, company_name.replace(" ", "_"))
    os.makedirs(company_dir, exist_ok=True)
    res = get_clean_10k(company_name, output_dir=company_dir)

    return {
        "company": company_name,
        "ticker": ticker,
        "files": {
            "10-K": res["file_path"]
        }
    }
