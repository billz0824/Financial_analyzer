from flask import Flask, render_template, request
import os
import uuid
from Execution.baseline_executer import execute_baseline_strategies

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for server environments
import matplotlib.pyplot as plt


app = Flask(__name__)
PLOT_FOLDER = "static"
os.makedirs(PLOT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    ticker = request.form["ticker"]
    start_date = request.form["start_date"]
    end_date = request.form["end_date"]
    test_ratio = float(request.form["test_ratio"])
    selected_models = request.form.getlist("models")

    # Run only selected models
    all_results = execute_baseline_strategies(ticker, start_date, end_date, test_ratio, selected_models)
    common_index = all_results["Buy and Hold"].index
    for label in all_results:
        all_results[label] = all_results[label].reindex(common_index)

    # Plot generation
    plot_filename = f"comparison_plot_{uuid.uuid4().hex}.png"
    plot_path = os.path.join(PLOT_FOLDER, plot_filename)

    plt.figure(figsize=(18, 10))
    for label, df in all_results.items():
        df = df.dropna(subset=["Funds"]).sort_index()
        plt.plot(df.index, df['Funds'], label=label)
    plt.title(f"Strategy Comparison for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Total Funds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Return statistics
    stats = {}
    for label, df in all_results.items():
        start_val = df['Funds'].iloc[0]
        end_val = df['Funds'].iloc[-1]
        stats[label] = f"{((end_val - start_val) / start_val) * 100:.2f}%"

    return render_template('results.html',
                           ticker=ticker,
                           start=start_date,
                           end=end_date,
                           plot_file=plot_filename,
                           stats=stats)

if __name__ == '__main__':
    app.run(debug=True)
