import os
import sys
import json
import re

# Add backend to path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, BACKEND_DIR)

from flask import Flask, request, jsonify, redirect, url_for, flash, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config import Config
from pdf_extractor import PDFExtractor
from ai_evaluator import AIEvaluator
from models import DatabaseManager

# ===== PATHS =====
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(PROJECT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===== APP =====
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = "my-secret-key-123"
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
CORS(app)

db = DatabaseManager()
extractor = PDFExtractor()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf"}


# ===== HTML PAGES =====

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Paper Evaluator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }
        .navbar {
            background: #1e293b;
            padding: 15px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
        }
        .navbar h1 {
            font-size: 1.5rem;
            color: #3b82f6;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        .hero {
            text-align: center;
            margin-bottom: 30px;
        }
        .hero h2 { font-size: 2rem; margin-bottom: 10px; }
        .hero p { color: #94a3b8; font-size: 1.1rem; }
        .upload-card {
            background: #1e293b;
            border-radius: 16px;
            padding: 40px;
            border: 1px solid #334155;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 25px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #cbd5e1;
        }
        .file-upload-area {
            border: 2px dashed #475569;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .file-upload-area:hover {
            border-color: #3b82f6;
            background: rgba(59, 130, 246, 0.05);
        }
        .upload-icon { font-size: 3rem; display: block; margin-bottom: 10px; }
        .browse-text { color: #3b82f6; font-weight: 600; }
        input[type="file"] {
            margin-top: 15px;
            color: #e2e8f0;
        }
        select {
            width: 100%;
            padding: 12px;
            background: #0f172a;
            border: 1px solid #475569;
            border-radius: 8px;
            color: #e2e8f0;
            font-size: 1rem;
        }
        .btn-primary {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
        }
        .alert {
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .alert-error {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            color: #fca5a5;
        }
        .alert-success {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid #22c55e;
            color: #86efac;
        }
        .loading {
            text-align: center;
            margin-top: 20px;
            display: none;
        }
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #334155;
            border-top: 4px solid #3b82f6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .history-section { margin-top: 40px; }
        .history-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .history-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            text-decoration: none;
            color: #e2e8f0;
            transition: all 0.3s;
            display: block;
        }
        .history-card:hover {
            border-color: #3b82f6;
            transform: translateY(-3px);
        }
        .history-details {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
        }
        .grade { color: #22c55e; font-weight: 700; font-size: 1.2rem; }
        .history-date { color: #64748b; font-size: 0.85rem; margin-top: 10px; }
        .format-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            margin-top: 15px;
        }
        .format-card pre {
            background: #0f172a;
            padding: 15px;
            border-radius: 8px;
            color: #94a3b8;
            margin-top: 10px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Paper Evaluator</h1>
        <a href="/" style="color:#94a3b8;text-decoration:none;">Home</a>
    </nav>

    <div class="container">
        {{FLASH_MESSAGES}}

        <div class="hero">
            <h2>Upload Answer Sheet for AI Evaluation</h2>
            <p>Upload a PDF with questions and answers. AI will evaluate each answer.</p>
        </div>

        <div class="upload-card">
            <form action="/upload" method="POST" enctype="multipart/form-data" id="uploadForm">
                <div class="form-group">
                    <label>Select PDF File</label>
                    <div class="file-upload-area">
                        <span class="upload-icon">PDF</span>
                        <p>Choose a PDF file or drag it here</p>
                        <input type="file" name="file" accept=".pdf" required>
                    </div>
                </div>

                <div class="form-group">
                    <label>Maximum Marks Per Question</label>
                    <select name="max_marks">
                        <option value="5">5 Marks</option>
                        <option value="10" selected>10 Marks</option>
                        <option value="15">15 Marks</option>
                        <option value="20">20 Marks</option>
                    </select>
                </div>

                <button type="submit" class="btn-primary" id="submitBtn">
                    Evaluate Paper
                </button>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>AI is evaluating your paper... Please wait.</p>
                </div>
            </form>
        </div>

        {{HISTORY}}

        <div class="format-card">
            <h3>PDF Format Example</h3>
            <pre>
Q1. What is Python?
Answer: Python is a high-level programming language.

Q2. Explain OOP concepts.
Answer: OOP includes encapsulation, inheritance,
polymorphism, and abstraction.
            </pre>
        </div>
    </div>

    <script>
        document.getElementById("uploadForm").addEventListener("submit", function() {
            document.getElementById("submitBtn").disabled = true;
            document.getElementById("submitBtn").textContent = "Evaluating...";
            document.getElementById("loading").style.display = "block";
        });
    </script>
</body>
</html>
"""

RESULTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results - AI Paper Evaluator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }
        .navbar {
            background: #1e293b;
            padding: 15px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
        }
        .navbar h1 { font-size: 1.5rem; color: #3b82f6; }
        .container { max-width: 900px; margin: 0 auto; padding: 30px 20px; }
        .results-header { text-align: center; margin-bottom: 30px; }
        .results-header h2 { font-size: 2rem; margin-bottom: 10px; }
        .filename { color: #94a3b8; }
        .summary-card {
            background: #1e293b;
            border-radius: 16px;
            padding: 30px;
            border: 1px solid #334155;
            margin-bottom: 30px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        .summary-item { text-align: center; }
        .summary-label { display: block; color: #94a3b8; font-size: 0.9rem; margin-bottom: 5px; }
        .summary-value { font-size: 1.8rem; font-weight: 700; color: #3b82f6; }
        .progress-container {
            width: 100%;
            height: 12px;
            background: #334155;
            border-radius: 6px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            border-radius: 6px;
        }
        .question-card {
            background: #1e293b;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid #334155;
            overflow: hidden;
        }
        .question-header {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px 20px;
            background: #334155;
        }
        .q-number {
            background: #3b82f6;
            color: white;
            padding: 5px 12px;
            border-radius: 8px;
            font-weight: 700;
        }
        .q-marks { font-weight: 600; flex: 1; }
        .q-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            background: rgba(59,130,246,0.2);
            color: #3b82f6;
        }
        .question-body { padding: 20px; }
        .q-section { margin-bottom: 15px; }
        .q-section p { color: #94a3b8; margin-top: 5px; line-height: 1.6; }
        .feedback {
            background: rgba(59,130,246,0.05);
            padding: 15px;
            border-radius: 8px;
        }
        .strengths {
            background: rgba(34,197,94,0.05);
            padding: 15px;
            border-radius: 8px;
        }
        .improvements {
            background: rgba(234,179,8,0.05);
            padding: 15px;
            border-radius: 8px;
        }
        .q-section ul { list-style: none; margin-top: 8px; }
        .q-section ul li { padding: 4px 0; color: #94a3b8; }
        .btn-back {
            display: inline-block;
            padding: 12px 30px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            margin-top: 20px;
        }
        .btn-back:hover { transform: translateY(-2px); }
        @media (max-width: 768px) {
            .summary-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Paper Evaluator</h1>
        <a href="/" style="color:#94a3b8;text-decoration:none;">Home</a>
    </nav>

    <div class="container">
        <div class="results-header">
            <h2>Evaluation Results</h2>
            <p class="filename">{{FILENAME}}</p>
        </div>

        <div class="summary-card">
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="summary-label">Total Marks</span>
                    <span class="summary-value">{{TOTAL_MARKS}} / {{MAX_MARKS}}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Percentage</span>
                    <span class="summary-value">{{PERCENTAGE}}%</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Grade</span>
                    <span class="summary-value" style="color:#22c55e;">{{GRADE}}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Questions</span>
                    <span class="summary-value">{{TOTAL_QUESTIONS}}</span>
                </div>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width:{{PERCENTAGE}}%"></div>
            </div>
        </div>

        <h3 style="margin-bottom:20px;">Question-wise Analysis</h3>
        {{QUESTIONS}}

        <a href="/" class="btn-back">Back to Home</a>
    </div>
</body>
</html>
"""


# ===== ROUTES =====

@app.route("/")
def index():
    flash_html = ""
    with app.test_request_context():
        pass

    # Build history
    history_html = ""
    try:
        evaluations = db.get_all_evaluations()
        if evaluations:
            history_html = '<div class="history-section"><h3>Previous Evaluations</h3><div class="history-grid">'
            for e in evaluations:
                history_html += f'''
                <a href="/results/{e.id}" class="history-card">
                    <div><strong>{e.filename}</strong></div>
                    <div class="history-details">
                        <span class="grade">{e.grade}</span>
                        <span>{e.percentage}%</span>
                    </div>
                    <div class="history-date">{e.upload_date.strftime("%d %b %Y, %I:%M %p")}</div>
                </a>
                '''
            history_html += '</div></div>'
    except Exception:
        pass

    html = INDEX_HTML.replace("{{FLASH_MESSAGES}}", flash_html)
    html = html.replace("{{HISTORY}}", history_html)
    return html


@app.route("/upload", methods=["POST"])
def upload_and_evaluate():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        return redirect(url_for("index"))

    try:
        max_marks = int(request.form.get("max_marks", 10))
    except ValueError:
        max_marks = 10

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        print(f"Extracting text from: {filename}")
        text = extractor.extract_text(filepath)

        if not text or len(text.strip()) < 10:
            return "<h1>Error: Could not extract text from PDF</h1><a href='/'>Go Back</a>"

        print(f"Extracted {len(text)} characters")

        qa_pairs = extractor.extract_questions_and_answers(text)

        if not qa_pairs:
            return "<h1>Error: Could not find Q&A in PDF</h1><a href='/'>Go Back</a>"

        print(f"Found {len(qa_pairs)} question(s)")

        evaluator = AIEvaluator(api_key=Config.GEMINI_API_KEY)
        results = evaluator.evaluate_full_paper(
            qa_pairs,
            max_marks_per_question=max_marks
        )
        print(f"Evaluation complete! Score: {results['percentage']}%")

        eval_id = db.save_evaluation(filename, results)
        print(f"Saved with ID: {eval_id}")

        return redirect(f"/results/{eval_id}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return f"<h1>Error: {str(e)}</h1><a href='/'>Go Back</a>"


@app.route("/results/<int:eval_id>")
def view_results(eval_id):
    evaluation = db.get_evaluation_by_id(eval_id)

    if not evaluation:
        return "<h1>Evaluation not found</h1><a href='/'>Go Back</a>"

    try:
        detailed_results = json.loads(evaluation.detailed_results)
    except Exception:
        detailed_results = []

    # Build questions HTML
    questions_html = ""
    for r in detailed_results:
        strengths_html = ""
        if r.get("strengths"):
            strengths_html = '<div class="q-section strengths"><strong>Strengths:</strong><ul>'
            for s in r["strengths"]:
                strengths_html += f"<li>{s}</li>"
            strengths_html += "</ul></div>"

        improvements_html = ""
        if r.get("improvements"):
            improvements_html = '<div class="q-section improvements"><strong>Areas for Improvement:</strong><ul>'
            for i in r["improvements"]:
                improvements_html += f"<li>{i}</li>"
            improvements_html += "</ul></div>"

        questions_html += f'''
        <div class="question-card">
            <div class="question-header">
                <span class="q-number">Q{r.get("question_no", "")}</span>
                <span class="q-marks">{r.get("marks_obtained", 0)} / {r.get("max_marks", 10)} marks</span>
                <span class="q-badge">{r.get("correctness", "N/A")}</span>
            </div>
            <div class="question-body">
                <div class="q-section">
                    <strong>Question:</strong>
                    <p>{r.get("question", "")}</p>
                </div>
                <div class="q-section">
                    <strong>Student Answer:</strong>
                    <p>{r.get("student_answer", "")}</p>
                </div>
                <div class="q-section feedback">
                    <strong>AI Feedback:</strong>
                    <p>{r.get("feedback", "")}</p>
                </div>
                {strengths_html}
                {improvements_html}
            </div>
        </div>
        '''

    html = RESULTS_HTML
    html = html.replace("{{FILENAME}}", evaluation.filename)
    html = html.replace("{{TOTAL_MARKS}}", str(evaluation.total_marks))
    html = html.replace("{{MAX_MARKS}}", str(evaluation.max_marks))
    html = html.replace("{{PERCENTAGE}}", str(evaluation.percentage))
    html = html.replace("{{GRADE}}", evaluation.grade)
    html = html.replace("{{TOTAL_QUESTIONS}}", str(evaluation.total_questions))
    html = html.replace("{{QUESTIONS}}", questions_html)

    return html


# ===== RUN =====
if __name__ == "__main__":
    print("=" * 55)
    print("AI Paper Evaluator Running!")
    print("Open: http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)