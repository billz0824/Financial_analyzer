from flask import Flask, render_template, request
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    ticker = request.form["ticker"]
    start_date = request.form["start_date"]
    end_date = request.form["end_date"]
    test_ratio = request.form["test_ratio"]
    algorithm_file = request.files.get("algorithm")

    # Save uploaded algorithm
    saved_filename = ""
    if algorithm_file:
        filename = algorithm_file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        algorithm_file.save(file_path)
        saved_filename = filename

    # Validate date format
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD."

    # TODO: Run the uploaded model file on the data from start_date to end_date
    # TODO: Compare with baseline strategies (e.g., buy-and-hold)
    # This is where you'd load `file_path`, import or run the logic (with sandboxing for safety), and return results

    result = f"""
    Ticker: {ticker}  
    Time Period: {start_date} to {end_date}  
    Uploaded model: {saved_filename}  
    Result: TODO - run backtest and return performance comparison.
    """

    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)