from flask import Flask, render_template, request
import random
import os

from PyPDF2 import PdfReader

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    company = request.form["company"]
    question = request.form["question"]
    file = request.files.get("file")

    extracted_text = ""

    if file:
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        if filename.endswith(".pdf"):
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                extracted_text = f.read()

    # TODO: Use `company`, `question`, and `extracted_text` for RAG or answer generation
    answer = "todo"
    sources = "sources"

    return render_template('results.html', company=company, question=question, answer=answer, sources=sources)

if __name__ == '__main__':
    app.run(debug=True)