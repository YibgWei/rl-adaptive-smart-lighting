""" Open: http://localhost:5000
"""

import os
import json
from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder="templates")

DATA_FILE = "episode_data.json"


def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    data = load_data()
    if data is None:
        return jsonify({
            "error": "episode_data.json not found",
            "hint": "Run: python export_episode.py"
        }), 404
    return jsonify(data)


@app.route("/api/status")
def api_status():
    exists = os.path.exists(DATA_FILE)
    algos  = []
    if exists:
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                d = json.load(f)
            algos = list(d.keys())
        except Exception:
            pass
    return jsonify({
        "data_file_exists": exists,
        "algorithms_available": algos,
        "data_file": DATA_FILE,
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Smart Lighting RL Simulation — Flask App")
    print("="*50)
    if not os.path.exists(DATA_FILE):
        print(f"\n  ⚠  WARNING: {DATA_FILE} not found!")
        print("  Run this first:  python export_episode.py\n")
    else:
        print(f"\n  ✓  {DATA_FILE} found")
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                d = json.load(f)
            print(f"  ✓  Algorithms: {list(d.keys())}")
        except Exception as e:
            print(f"  ✗  Error reading data: {e}")
    print("\n  🚀 Starting at http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
