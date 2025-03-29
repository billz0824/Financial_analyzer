from sec_edgar_downloader import Downloader
import os
import requests
from bs4 import BeautifulSoup, Comment
from sec_edgar_downloader import Downloader
from .info import EMAIL, HEADERS
from .data_tools import get_ticker
import re
from weasyprint import HTML


def get_cik(ticker: str) -> str:
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany"
    response = requests.get(url, headers=HEADERS)
    match = re.search(r'CIK=(\d{10})', response.text)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"CIK not found for ticker: {ticker}")

def download_latest_10k(ticker, output_dir, email=EMAIL):
    dl = Downloader(
        company_name=ticker,
        download_folder=output_dir,
        email_address=email)
    dl.get("10-K", ticker, limit=1)

    filing_dir = os.path.join(output_dir, "sec-edgar-filings", ticker, "10-K")

    if not os.path.exists(filing_dir):
        raise FileNotFoundError(f"Directory not found: {filing_dir}")
    subdirs = sorted(os.listdir(filing_dir), reverse=True)

    if not subdirs:
        raise FileNotFoundError("No 10-K downloaded.")

    return subdirs[0]

def build_filing_index_url(accession_number, cik):
    accession_clean = accession_number.replace("-", "")
    s = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession_number}-index.html"
    return s

def convert_ix_to_raw_htm_url(ix_url: str) -> str:
    """
    Convert an ix?doc= URL to the raw .htm SEC filing URL.
    """
    if "ix?doc=" not in ix_url:
        raise ValueError("This is not a valid ix?doc= URL.")

    # Strip the ix?doc= prefix
    raw_path = ix_url.split("ix?doc=")[-1]

    # Join with base SEC URL
    return f"https://www.sec.gov{raw_path}"


def get_10k_html_url(index_url):
    res = requests.get(index_url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    table = soup.find("table", class_="tableFile")
    if not table:
        raise ValueError("Couldn't find the document table.")

    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) >= 4:
            doc_type = cols[3].text.strip().lower()
            doc_link = cols[2].find("a")["href"]

            # Actual 10-K doc — use ix?doc= wrapper
            if "10-k" in doc_type and doc_link.endswith(".htm"):
                url = f"https://www.sec.gov{doc_link}"
                url = convert_ix_to_raw_htm_url(url)
                return url

    raise ValueError("Could not find a 10-K filing with a .htm document.")


def download_htm(url: str, save_path: str):
    response = requests.get(url, headers=HEADERS)

    file_path = os.path.join(save_path, "10-K")
    pdf_path = os.path.join(save_path, "10-K.pdf")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    HTML(file_path).write_pdf(pdf_path)
    
    return file_path


def extract_clean_10k_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove garbage: scripts, styles, comments, XBRL tags
    for tag in soup(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    for tag in soup.find_all(True):
        if tag.name.startswith("ix:"):
            tag.unwrap()
        for attr in list(tag.attrs):
            if attr.startswith("ix:"):
                del tag.attrs[attr]

    # Flatten tables
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            rows.append("\t".join(cols))
        table.replace_with("\n".join(rows))

    # Convert to text
    raw_text = soup.get_text(separator="\n")

    # Collapse excessive spacing
    clean_text = re.sub(r"\n\s*\n", "\n\n", raw_text)
    clean_text = re.sub(r"[ \t]+", " ", clean_text)

    return clean_text.strip()


def get_clean_10k(company_name, output_dir="data/sec", email=EMAIL):
    print(f"🔎 Fetching 10-K for {company_name}...")

    # Get ticker and CIK
    ticker = get_ticker(company_name)
    cik = get_cik(ticker)

    # Download the latest 10-K filing
    accession_number = download_latest_10k(ticker, output_dir, email)

    # Get the URL to the filing index and then the 10-K HTML doc
    index_url = build_filing_index_url(accession_number, cik)
    doc_url = get_10k_html_url(index_url)
    print(doc_url)

    # Download htm file
    path = download_htm(doc_url, save_path=output_dir)
    print(path)

    # Extract and clean the text from the 10-K HTML
    clean_text = extract_clean_10k_text(path)

    # Save file
    company_dir = os.path.join(output_dir, company_name.replace(" ", "_"))
    os.makedirs(company_dir, exist_ok=True)
    file_path = os.path.join(company_dir, "10-K.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"Saved clean 10-K text to {file_path}")
    return {
        "company": company_name,
        "ticker": ticker,
        "cik": cik,
        "accession": accession_number,
        "document_url": doc_url,
        "file_path": file_path,
        "clean_text": clean_text
    }
