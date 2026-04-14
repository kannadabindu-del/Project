import os
import json
import re
import numpy as np
from flask import Flask, request, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ==================== IMPORTS ====================

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    print("PyTorch loaded")
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
    print("SentenceTransformer loaded")
except ImportError:
    HAS_ST = False
    print("SentenceTransformer not available")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(PROJECT_DIR, "uploads")
DB_FILE = os.path.join(PROJECT_DIR, "evaluations.json")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "secret123"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)

# ==================== FIND WORKING GEMINI MODEL ====================

GEMINI_MODEL = None

def find_gemini_model():
    global GEMINI_MODEL
    if not HAS_GENAI or not GEMINI_API_KEY:
        print("Gemini not configured")
        return None

    genai.configure(api_key=GEMINI_API_KEY)

    # Try models in order of preference
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-pro",
    ]

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say OK")
            if response:
                print(f"Gemini model working: {model_name}")
                GEMINI_MODEL = model_name
                return model_name
        except Exception as e:
            print(f"Model {model_name} failed: {str(e)[:50]}")
            continue

    # Try listing available models
    try:
        available = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                available.append(m.name)
                print(f"Available model: {m.name}")

        if available:
            model_name = available[0].replace("models/", "")
            GEMINI_MODEL = model_name
            print(f"Using model: {model_name}")
            return model_name
    except Exception as e:
        print(f"Could not list models: {e}")

    print("No working Gemini model found")
    return None


# ==================== RNN MODEL ====================

class AnswerScorerRNN(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_layers=3):
        super(AnswerScorerRNN, self).__init__()

        self.question_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.answer_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        combined_dim = hidden_dim * 2 * 4

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(self, embedding, rnn):
        x = embedding.unsqueeze(0).unsqueeze(0)
        out, (hidden, _) = rnn(x)
        final = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return final.squeeze(0)

    def forward(self, q_emb, a_emb):
        q_enc = self.encode(q_emb, self.question_encoder)
        a_enc = self.encode(a_emb, self.answer_encoder)

        if q_enc.dim() == 1:
            q_enc = q_enc.unsqueeze(0)
        if a_enc.dim() == 1:
            a_enc = a_enc.unsqueeze(0)

        combined = torch.cat([
            q_enc,
            a_enc,
            q_enc * a_enc,
            torch.abs(q_enc - a_enc)
        ], dim=-1)

        score = self.classifier(combined)
        return score.squeeze()


# ==================== DEEP LEARNING EVALUATOR ====================

class HybridEvaluator:
    def __init__(self, api_key=""):
        self.api_key = api_key
        self.embedding_model = None
        self.rnn_model = None
        self.gemini_model_name = None

        # Load sentence transformer
        if HAS_ST:
            try:
                print("Loading Sentence Transformer...")
                self.embedding_model = SentenceTransformer(
                    'all-MiniLM-L6-v2'
                )
                print("Sentence Transformer ready!")
            except Exception as e:
                print(f"Transformer error: {e}")

        # Initialize RNN
        if HAS_TORCH:
            try:
                self.rnn_model = AnswerScorerRNN(
                    input_dim=384,
                    hidden_dim=256,
                    num_layers=3
                )
                self.rnn_model.eval()
                print("RNN Model ready!")
            except Exception as e:
                print(f"RNN error: {e}")

        # Find working Gemini model
        self.gemini_model_name = find_gemini_model()

    def get_embedding(self, text):
        if self.embedding_model:
            try:
                emb = self.embedding_model.encode(
                    text,
                    convert_to_numpy=True
                )
                return emb
            except Exception as e:
                print(f"Embedding error: {e}")
        return None

    def semantic_similarity(self, q_emb, a_emb):
        try:
            if q_emb is not None and a_emb is not None:
                sim = cosine_similarity(
                    q_emb.reshape(1, -1),
                    a_emb.reshape(1, -1)
                )[0][0]
                return float(max(0, min(1, sim)))
        except Exception as e:
            print(f"Similarity error: {e}")
        return 0.5

    def keyword_score(self, question, answer):
        try:
            stop_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you',
                'all', 'can', 'was', 'one', 'our', 'out', 'has',
                'have', 'with', 'this', 'that', 'what', 'from',
                'they', 'will', 'been', 'said', 'some', 'would',
                'make', 'like', 'into', 'time', 'when', 'explain',
                'define', 'describe', 'discuss', 'what', 'which',
                'how', 'why', 'give', 'name', 'list', 'write',
                'different', 'types', 'example', 'examples'
            }

            q_words = set(
                re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
            ) - stop_words

            a_words = set(
                re.findall(r'\b[a-zA-Z]{3,}\b', answer.lower())
            ) - stop_words

            if not q_words:
                return 0.5

            overlap = len(q_words & a_words) / len(q_words)

            # TF-IDF similarity
            try:
                if HAS_SKLEARN:
                    vec = TfidfVectorizer(stop_words='english')
                    tfidf = vec.fit_transform([question, answer])
                    tfidf_sim = cosine_similarity(
                        tfidf[0:1], tfidf[1:2]
                    )[0][0]
                    overlap = (overlap * 0.5) + (tfidf_sim * 0.5)
            except Exception:
                pass

            return float(max(0, min(1, overlap)))
        except Exception:
            return 0.3

    def answer_quality(self, answer, max_marks):
        word_count = len(answer.split())
        words = re.findall(r'\b[a-zA-Z]+\b', answer.lower())
        unique_words = len(set(words))
        sentences = len(re.split(r'[.!?]+', answer))

        # Expected length based on marks
        if max_marks <= 5:
            exp = 50
        elif max_marks <= 10:
            exp = 100
        elif max_marks <= 20:
            exp = 200
        elif max_marks <= 50:
            exp = 350
        else:
            exp = 500

        length_score = min(1.0, word_count / exp)
        diversity = (
            unique_words / len(words) if words else 0
        )

        # Content quality checks
        has_def = bool(re.search(
            r'(is|are|means|defined|refers|stands for)',
            answer.lower()
        ))
        has_ex = bool(re.search(
            r'(example|such as|for instance|like|e\.g)',
            answer.lower()
        ))
        has_enum = bool(re.search(
            r'(\d+\.|first|second|third|finally|additionally)',
            answer.lower()
        ))

        struct_score = 0.4
        if has_def:
            struct_score += 0.2
        if has_ex:
            struct_score += 0.2
        if has_enum:
            struct_score += 0.2

        return {
            "score": (
                length_score * 0.4 +
                min(1.0, diversity) * 0.2 +
                struct_score * 0.4
            ),
            "word_count": word_count,
            "has_definition": has_def,
            "has_examples": has_ex,
            "has_enumeration": has_enum
        }

    def rnn_score(self, q_emb, a_emb):
        if not HAS_TORCH or self.rnn_model is None:
            return None
        try:
            q_t = torch.FloatTensor(q_emb)
            a_t = torch.FloatTensor(a_emb)
            with torch.no_grad():
                score = self.rnn_model(q_t, a_t)
            return float(score.item())
        except Exception as e:
            print(f"RNN error: {e}")
            return None

    def gemini_evaluate(self, question, answer, max_marks):
        if not HAS_GENAI or not self.api_key:
            return None
        if not self.gemini_model_name:
            self.gemini_model_name = find_gemini_model()
        if not self.gemini_model_name:
            return None

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.gemini_model_name)

            prompt = f"""You are a strict university professor evaluating an exam answer.

QUESTION: {question}

STUDENT ANSWER: {answer}

MAXIMUM MARKS: {max_marks}

EVALUATE STRICTLY AND FAIRLY:

Give marks based on these criteria:
- If answer covers ALL key points correctly: 85-100% of marks
- If answer covers MOST key points: 70-84% of marks  
- If answer covers SOME key points: 50-69% of marks
- If answer is PARTIALLY correct: 30-49% of marks
- If answer is MOSTLY WRONG: 10-29% of marks
- If answer is BLANK or COMPLETELY WRONG: 0-9% of marks

IMPORTANT:
- A correct and complete answer MUST get high marks
- Do NOT give 0 marks if student wrote something relevant
- Be ACCURATE - good answers deserve good marks
- Consider depth and accuracy of explanation

Return ONLY this JSON, no other text:
{{
    "marks_obtained": <number from 0 to {max_marks}>,
    "max_marks": {max_marks},
    "percentage": <marks_obtained / {max_marks} * 100>,
    "feedback": "<3-4 sentences: what was correct, what was missing, overall assessment>",
    "strengths": ["<specific strength 1>", "<specific strength 2>", "<specific strength 3>"],
    "improvements": ["<specific gap 1>", "<specific gap 2>"],
    "correctness": "<Excellent or Good or Average or Poor>",
    "completeness": "<Excellent or Good or Average or Poor>",
    "key_points_covered": "<brief list of main points student covered>"
}}"""

            for attempt in range(3):
                try:
                    response = model.generate_content(prompt)
                    text = response.text.strip()
                    text = re.sub(r'```json\s*', '', text)
                    text = re.sub(r'```\s*', '', text)
                    text = text.strip()

                    match = re.search(r'\{.*\}', text, re.DOTALL)
                    if match:
                        result = json.loads(match.group())
                        marks = float(
                            result.get("marks_obtained", 0)
                        )
                        marks = max(0, min(max_marks, marks))
                        result["marks_obtained"] = round(marks, 1)
                        result["percentage"] = round(
                            (marks / max_marks) * 100, 1
                        )
                        return result
                except json.JSONDecodeError:
                    if attempt == 2:
                        try:
                            m = re.search(
                                r'"marks_obtained"\s*:\s*(\d+(?:\.\d+)?)',
                                text
                            )
                            if m:
                                marks = min(
                                    float(m.group(1)), max_marks
                                )
                                return {
                                    "marks_obtained": marks,
                                    "max_marks": max_marks,
                                    "percentage": round(
                                        marks / max_marks * 100, 1
                                    ),
                                    "feedback": "Evaluated successfully.",
                                    "strengths": ["Answer attempted"],
                                    "improvements": [],
                                    "correctness": "Average",
                                    "completeness": "Average",
                                    "key_points_covered": ""
                                }
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Gemini attempt {attempt+1} error: {e}")

        except Exception as e:
            print(f"Gemini error: {e}")

        return None

    def evaluate(self, question, answer, max_marks=10):
        print(f"  Words: {len(answer.split())}")

        # Blank answer check
        if (not answer or
                answer.strip() == "" or
                answer == "No answer provided" or
                len(answer.strip()) < 5):
            return {
                "marks_obtained": 0,
                "max_marks": max_marks,
                "percentage": 0.0,
                "feedback": "No answer was provided.",
                "strengths": [],
                "improvements": [
                    "Student did not attempt this question",
                    "Must study this topic"
                ],
                "correctness": "Poor",
                "completeness": "Poor",
                "score_breakdown": {}
            }

        # Get embeddings
        q_emb = self.get_embedding(question)
        a_emb = self.get_embedding(answer)

        scores = {}

        # 1. Semantic similarity
        if q_emb is not None and a_emb is not None:
            sem = self.semantic_similarity(q_emb, a_emb)
        else:
            sem = 0.5
        scores["semantic"] = sem
        print(f"  Semantic: {sem:.3f}")

        # 2. Keyword score
        kw = self.keyword_score(question, answer)
        scores["keyword"] = kw
        print(f"  Keywords: {kw:.3f}")

        # 3. Answer quality
        quality = self.answer_quality(answer, max_marks)
        scores["quality"] = quality["score"]
        print(f"  Quality:  {quality['score']:.3f}")

        # 4. RNN score
        rnn = None
        if q_emb is not None and a_emb is not None:
            rnn = self.rnn_score(q_emb, a_emb)
        scores["rnn"] = rnn
        print(f"  RNN:      {rnn:.3f}" if rnn else "  RNN:      N/A")

        # 5. Gemini AI evaluation (most accurate)
        print("  Calling Gemini AI...")
        gemini = self.gemini_evaluate(question, answer, max_marks)
        if gemini:
            gemini_pct = gemini.get("percentage", 50) / 100
            scores["gemini"] = gemini_pct
            print(f"  Gemini:   {gemini_pct:.3f}")
        else:
            scores["gemini"] = None
            print("  Gemini:   Not available")

        # ==================== WEIGHTED SCORING ====================

        if scores["gemini"] is not None:
            # Gemini available - use it as primary score
            # Gemini: 60%, Semantic: 20%, Quality: 10%, Keywords: 5%, RNN: 5%

            final = (
                scores["gemini"] * 0.60 +
                scores["semantic"] * 0.20 +
                scores["quality"] * 0.10 +
                scores["keyword"] * 0.05
            )

            if rnn is not None:
                # Calibrate RNN using semantic as reference
                calibrated_rnn = (
                    rnn * 0.3 + scores["semantic"] * 0.7
                )
                final = final * 0.95 + calibrated_rnn * 0.05

        else:
            # Gemini NOT available - use semantic as primary
            # Semantic is very accurate for relevance scoring

            sem_score = scores["semantic"]
            kw_score = scores["keyword"]
            q_score = scores["quality"]

            # Semantic-based scoring with calibration
            if sem_score >= 0.88:
                base = 0.90
            elif sem_score >= 0.82:
                base = 0.82
            elif sem_score >= 0.75:
                base = 0.74
            elif sem_score >= 0.68:
                base = 0.65
            elif sem_score >= 0.60:
                base = 0.56
            elif sem_score >= 0.50:
                base = 0.47
            else:
                base = 0.35

            final = (
                base * 0.55 +
                kw_score * 0.20 +
                q_score * 0.20 +
                sem_score * 0.05
            )

            # Quality bonus
            if quality["has_definition"] and quality["has_examples"]:
                final = min(1.0, final * 1.08)
            elif quality["has_definition"] or quality["has_examples"]:
                final = min(1.0, final * 1.04)

        final = max(0.0, min(1.0, final))

        # Calculate marks
        marks = round(final * max_marks, 1)

        print(f"  FINAL:    {marks}/{max_marks} ({final*100:.1f}%)")

        # Correctness label
        if final >= 0.85:
            correctness = "Excellent"
        elif final >= 0.65:
            correctness = "Good"
        elif final >= 0.45:
            correctness = "Average"
        else:
            correctness = "Poor"

        # Completeness label
        if quality["score"] >= 0.70:
            completeness = "Excellent"
        elif quality["score"] >= 0.55:
            completeness = "Good"
        elif quality["score"] >= 0.35:
            completeness = "Average"
        else:
            completeness = "Poor"

        # Use Gemini feedback if available
        if gemini:
            feedback = gemini.get("feedback", "")
            strengths = gemini.get("strengths", [])
            improvements = gemini.get("improvements", [])
            correctness = gemini.get("correctness", correctness)
            completeness = gemini.get("completeness", completeness)
        else:
            # Generate feedback from scores
            if final >= 0.85:
                feedback = (
                    "Excellent answer demonstrating thorough "
                    "understanding. The student has covered the "
                    "key concepts accurately and with good depth. "
                    "Very well structured response."
                )
            elif final >= 0.70:
                feedback = (
                    "Good answer covering most important points. "
                    "The core concepts are correct. Some additional "
                    "detail or examples would make this answer even "
                    "stronger."
                )
            elif final >= 0.55:
                feedback = (
                    "Satisfactory answer with correct basic "
                    "understanding. Several key points are present "
                    "but the answer lacks depth and completeness. "
                    "More detailed explanation needed."
                )
            elif final >= 0.40:
                feedback = (
                    "Partial answer showing some knowledge of the "
                    "topic. Significant gaps in coverage and depth. "
                    "Student needs to study this topic more "
                    "thoroughly."
                )
            else:
                feedback = (
                    "Answer shows limited understanding of the "
                    "topic. Major concepts are missing or incorrect. "
                    "Significant revision of this topic is required."
                )

            strengths = []
            if sem_score > 0.70:
                strengths.append("Answer is relevant to the question")
            if quality["word_count"] > 50:
                strengths.append("Good level of detail provided")
            if quality["has_definition"]:
                strengths.append("Includes clear definitions")
            if quality["has_examples"]:
                strengths.append("Provides helpful examples")
            if quality["has_enumeration"]:
                strengths.append("Well structured with numbered points")
            if kw_score > 0.5:
                strengths.append("Uses correct technical terminology")
            if not strengths:
                strengths.append("Question was attempted")

            improvements = []
            if quality["word_count"] < 40:
                improvements.append("Expand the answer with more detail")
            if not quality["has_examples"]:
                improvements.append("Include specific examples")
            if not quality["has_definition"]:
                improvements.append("Add precise definitions")
            if kw_score < 0.3:
                improvements.append("Use more subject-specific terms")
            if sem_score < 0.70:
                improvements.append("Focus more on answering directly")
            if not improvements:
                improvements.append("Minor improvements in depth")

        return {
            "marks_obtained": marks,
            "max_marks": max_marks,
            "percentage": round(final * 100, 1),
            "feedback": feedback,
            "strengths": strengths[:4],
            "improvements": improvements[:4],
            "correctness": correctness,
            "completeness": completeness,
            "score_breakdown": {
                "semantic": round(sem * 100, 1),
                "keywords": round(kw * 100, 1),
                "quality": round(quality["score"] * 100, 1),
                "rnn": round(rnn * 100, 1) if rnn else "N/A",
                "gemini": (
                    round(scores["gemini"] * 100, 1)
                    if scores["gemini"] is not None else "N/A"
                )
            }
        }


# ==================== INIT EVALUATOR ====================

print("\nInitializing Hybrid Deep Learning Evaluator...")
EVALUATOR = HybridEvaluator(api_key=GEMINI_API_KEY)
print("Evaluator ready!\n")


# ==================== DATABASE ====================

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return []


def save_evaluation(q_file, a_file, results):
    db = load_db()
    eval_id = len(db) + 1
    db.append({
        "id": eval_id,
        "question_file": q_file,
        "answer_file": a_file,
        "total_marks": results["total_marks"],
        "max_marks": results["total_max_marks"],
        "percentage": results["percentage"],
        "grade": results["grade"],
        "grade_description": results["grade_description"],
        "total_questions": results["total_questions"],
        "results": results["results"]
    })
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)
    return eval_id


def get_evaluation(eval_id):
    for e in load_db():
        if e["id"] == eval_id:
            return e
    return None


# ==================== PDF EXTRACTION ====================

def extract_text(pdf_path):
    text = ""
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n\n"
        except Exception as e:
            print(f"pdfplumber: {e}")
    if not text.strip() and HAS_PYPDF2:
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n\n"
        except Exception as e:
            print(f"PyPDF2: {e}")
    return text.strip()


def extract_questions(text):
    questions = []
    p1 = re.findall(
        r'[Qq](?:uestion)?\s*(\d+)[.):\s]\s*(.*?)(?=\n\s*[Qq](?:uestion)?\s*\d+[.):\s]|$)',
        text, re.DOTALL)
    if p1:
        for m in p1:
            q = re.sub(r'\s+', ' ', m[1]).strip()
            if q:
                questions.append({
                    "question_no": int(m[0]),
                    "question": q
                })
        if questions:
            return questions

    p2 = re.findall(
        r'(?:^|\n)\s*(\d+)[.)]\s*(.*?)(?=\n\s*\d+[.)]|$)',
        text, re.DOTALL)
    if p2:
        for m in p2:
            q = re.sub(r'\s+', ' ', m[1]).strip()
            if q and len(q) > 5:
                questions.append({
                    "question_no": int(m[0]),
                    "question": q
                })
        if questions:
            return questions

    sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    for i, s in enumerate(sections):
        questions.append({
            "question_no": i + 1,
            "question": re.sub(r'\s+', ' ', s)
        })
    return questions


def extract_answers(text):
    answers = []

    p1 = re.findall(
        r'[Qq](?:uestion)?\s*(\d+)[.):\s]\s*(.*?)(?=\n\s*[Qq](?:uestion)?\s*\d+[.):\s]|$)',
        text, re.DOTALL)
    if p1:
        for m in p1:
            a = re.sub(r'\s+', ' ', m[1]).strip()
            if a:
                answers.append({
                    "question_no": int(m[0]),
                    "answer": a
                })
        if answers:
            return answers

    p2 = re.findall(
        r'[Aa](?:nswer|ns)?\s*(\d+)[.):\s]\s*(.*?)(?=\n\s*[Aa](?:nswer|ns)?\s*\d+[.):\s]|$)',
        text, re.DOTALL)
    if p2:
        for m in p2:
            a = re.sub(r'\s+', ' ', m[1]).strip()
            if a:
                answers.append({
                    "question_no": int(m[0]),
                    "answer": a
                })
        if answers:
            return answers

    p3 = re.findall(
        r'(?:^|\n)\s*(\d+)[.)]\s*(.*?)(?=\n\s*\d+[.)]|$)',
        text, re.DOTALL)
    if p3:
        for m in p3:
            a = re.sub(r'\s+', ' ', m[1]).strip()
            if a and len(a) > 5:
                answers.append({
                    "question_no": int(m[0]),
                    "answer": a
                })
        if answers:
            return answers

    sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    for i, s in enumerate(sections):
        answers.append({
            "question_no": i + 1,
            "answer": re.sub(r'\s+', ' ', s)
        })
    return answers


def match_qa(questions, answers):
    ans_dict = {a["question_no"]: a["answer"] for a in answers}
    return [
        {
            "question_no": q["question_no"],
            "question": q["question"],
            "answer": ans_dict.get(
                q["question_no"], "No answer provided"
            )
        }
        for q in questions
    ]


# ==================== GRADE CALCULATOR ====================

def calculate_grade(percentage):
    """
    Accurate grade calculation with descriptions
    """
    if percentage >= 95:
        return "A+", "Outstanding - Exceptional performance"
    elif percentage >= 90:
        return "A+", "Excellent - Near perfect performance"
    elif percentage >= 85:
        return "A", "Excellent - Very strong performance"
    elif percentage >= 80:
        return "A", "Very Good - Strong performance"
    elif percentage >= 75:
        return "B+", "Good - Above average performance"
    elif percentage >= 70:
        return "B+", "Good - Solid performance"
    elif percentage >= 65:
        return "B", "Above Average - Satisfactory performance"
    elif percentage >= 60:
        return "B", "Average - Acceptable performance"
    elif percentage >= 55:
        return "C+", "Below Average - Needs improvement"
    elif percentage >= 50:
        return "C", "Pass - Minimum acceptable performance"
    elif percentage >= 45:
        return "C-", "Marginal Pass - Needs significant improvement"
    elif percentage >= 40:
        return "D", "Poor - Barely passing"
    elif percentage >= 33:
        return "E", "Very Poor - Below minimum standard"
    else:
        return "F", "Fail - Did not meet minimum requirements"


# ==================== EVALUATE PAPER ====================

def evaluate_paper(qa_pairs, max_marks=10):
    results = []
    total = 0.0

    print(f"\nEvaluating {len(qa_pairs)} questions")
    print(f"Max marks per question: {max_marks}")
    print("=" * 55)

    for qa in qa_pairs:
        qno = qa["question_no"]
        print(f"\nQ{qno}: {qa['question'][:55]}...")

        r = EVALUATOR.evaluate(
            qa["question"],
            qa["answer"],
            max_marks
        )
        r["question_no"] = qno
        r["question"] = qa["question"]
        r["student_answer"] = qa["answer"]
        results.append(r)
        total += r.get("marks_obtained", 0)

    max_total = len(qa_pairs) * max_marks
    pct = round((total / max_total) * 100, 2) if max_total > 0 else 0
    grade, grade_desc = calculate_grade(pct)

    print("\n" + "=" * 55)
    print(f"TOTAL: {round(total,1)}/{max_total}")
    print(f"PERCENTAGE: {pct}%")
    print(f"GRADE: {grade} - {grade_desc}")
    print("=" * 55)

    return {
        "total_marks": round(total, 1),
        "total_max_marks": max_total,
        "percentage": pct,
        "total_questions": len(qa_pairs),
        "grade": grade,
        "grade_description": grade_desc,
        "results": results
    }


# ==================== CSS ====================

def get_css():
    return """
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:'Segoe UI',Arial,sans-serif;
    background:#0f172a;color:#e2e8f0;min-height:100vh}
    .nav{background:#1e293b;padding:15px 40px;
    border-bottom:1px solid #334155;display:flex;
    justify-content:space-between;align-items:center}
    .nav h1{color:#3b82f6;font-size:1.4rem;font-weight:700}
    .nav a{color:#94a3b8;text-decoration:none}
    .nav a:hover{color:#e2e8f0}
    .container{max-width:960px;margin:0 auto;padding:40px 20px}
    .hero{text-align:center;margin-bottom:35px}
    .hero h2{font-size:2rem;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px}
    .hero p{color:#94a3b8;font-size:1rem}
    .card{background:#1e293b;border-radius:16px;padding:35px;
    border:1px solid #334155;margin-bottom:25px}
    .upload-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:25px}
    .upload-box{border:2px dashed #475569;border-radius:12px;
    padding:30px 20px;text-align:center;transition:all 0.3s}
    .question-box{border-color:#3b82f6;background:rgba(59,130,246,0.04)}
    .answer-box{border-color:#8b5cf6;background:rgba(139,92,246,0.04)}
    .upload-box h3{color:#cbd5e1;margin-bottom:8px}
    .upload-box p{color:#64748b;font-size:0.85rem;margin-bottom:12px}
    .file-tag{display:inline-block;padding:4px 12px;border-radius:20px;
    font-size:0.8rem;font-weight:700;margin-bottom:10px}
    .q-tag{background:rgba(59,130,246,0.2);color:#93c5fd}
    .a-tag{background:rgba(139,92,246,0.2);color:#c4b5fd}
    input[type=file]{color:#e2e8f0;background:#0f172a;
    padding:8px;border-radius:8px;border:1px solid #334155;
    width:100%;cursor:pointer;font-size:0.85rem}
    .file-sel{color:#22c55e;font-size:0.85rem;margin-top:8px;
    font-weight:500;min-height:18px}
    .form-group{margin-bottom:20px}
    label{display:block;margin-bottom:8px;font-weight:600;color:#cbd5e1}
    select{width:100%;padding:12px;background:#0f172a;
    border:1px solid #475569;border-radius:8px;
    color:#e2e8f0;font-size:1rem}
    .info-box{background:rgba(59,130,246,0.06);
    border:1px solid rgba(59,130,246,0.25);
    border-radius:10px;padding:15px 20px;margin-bottom:20px;
    color:#93c5fd;font-size:0.88rem;line-height:1.8}
    .btn{width:100%;padding:16px;
    background:linear-gradient(135deg,#3b82f6,#8b5cf6);
    color:white;border:none;border-radius:12px;
    font-size:1.1rem;font-weight:700;cursor:pointer;transition:all 0.3s}
    .btn:hover{transform:translateY(-2px);
    box-shadow:0 10px 25px rgba(59,130,246,0.3)}
    .btn:disabled{opacity:0.7;cursor:not-allowed;transform:none}
    .loading{display:none;text-align:center;margin-top:25px;padding:20px}
    .spinner{width:50px;height:50px;border:4px solid #334155;
    border-top:4px solid #3b82f6;border-radius:50%;
    animation:spin 1s linear infinite;margin:0 auto 15px}
    @keyframes spin{to{transform:rotate(360deg)}}
    .loading p{color:#94a3b8}
    .steps{display:grid;grid-template-columns:repeat(3,1fr);
    gap:15px;margin-bottom:25px}
    .step{background:#0f172a;border-radius:10px;padding:15px;
    text-align:center;border:1px solid #334155}
    .step-num{width:32px;height:32px;background:#3b82f6;
    border-radius:50%;display:flex;align-items:center;
    justify-content:center;margin:0 auto 8px;font-weight:700}
    .step h4{color:#cbd5e1;font-size:0.9rem;margin-bottom:5px}
    .step p{color:#64748b;font-size:0.82rem}
    .summary-grid{display:grid;grid-template-columns:repeat(4,1fr);
    gap:15px;margin-bottom:25px}
    .stat{text-align:center}
    .stat-label{display:block;color:#94a3b8;font-size:0.75rem;
    margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em}
    .stat-val{font-size:1.8rem;font-weight:700}
    .prog-bg{width:100%;height:16px;background:#334155;
    border-radius:8px;overflow:hidden;margin-bottom:8px}
    .prog-fill{height:100%;
    background:linear-gradient(90deg,#3b82f6,#8b5cf6);
    border-radius:8px;transition:width 1s ease}
    .grade-box{text-align:center;padding:20px;
    background:#0f172a;border-radius:12px;margin-top:15px}
    .grade-letter{font-size:4rem;font-weight:900;
    line-height:1;margin-bottom:5px}
    .grade-desc{color:#94a3b8;font-size:0.9rem}
    .q-card{background:#1e293b;border-radius:12px;
    margin-bottom:20px;border:1px solid #334155;overflow:hidden}
    .q-head{display:flex;align-items:center;gap:12px;
    padding:14px 20px;background:#1e3a5f;flex-wrap:wrap}
    .q-num{background:#3b82f6;color:white;padding:5px 12px;
    border-radius:8px;font-weight:700;font-size:0.9rem}
    .q-body{padding:20px}
    .q-section{margin-bottom:14px}
    .q-section strong{color:#cbd5e1}
    .q-section p{color:#94a3b8;margin-top:5px;line-height:1.7}
    .feedback-box{background:rgba(59,130,246,0.07);padding:15px;
    border-radius:8px;border-left:3px solid #3b82f6;margin-top:10px}
    .strength-box{background:rgba(34,197,94,0.07);padding:15px;
    border-radius:8px;border-left:3px solid #22c55e;margin-top:10px}
    .improve-box{background:rgba(234,179,8,0.07);padding:15px;
    border-radius:8px;border-left:3px solid #eab308;margin-top:10px}
    .score-box{background:rgba(139,92,246,0.07);padding:15px;
    border-radius:8px;border-left:3px solid #8b5cf6;margin-top:10px}
    ul{list-style:none;margin-top:8px}
    ul li{padding:4px 0;color:#94a3b8;line-height:1.5}
    .score-pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
    .pill{background:#0f172a;padding:6px 12px;border-radius:8px;
    font-size:0.78rem;color:#94a3b8;border:1px solid #334155}
    .pill span{color:#3b82f6;font-weight:700;margin-left:4px}
    .pill.good span{color:#22c55e}
    .pill.avg span{color:#eab308}
    .pill.poor span{color:#ef4444}
    .badge{padding:4px 12px;border-radius:20px;font-size:0.82rem;
    font-weight:600;margin-left:auto}
    .badge-excellent{background:rgba(34,197,94,0.2);color:#22c55e}
    .badge-good{background:rgba(59,130,246,0.2);color:#93c5fd}
    .badge-average{background:rgba(234,179,8,0.2);color:#eab308}
    .badge-poor{background:rgba(239,68,68,0.2);color:#ef4444}
    .h-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));
    gap:15px;margin-top:15px}
    .h-card{background:#1e293b;border:1px solid #334155;
    border-radius:12px;padding:18px;text-decoration:none;
    color:#e2e8f0;display:block;transition:all 0.3s}
    .h-card:hover{border-color:#3b82f6;transform:translateY(-3px)}
    .back-btn{display:inline-block;padding:13px 35px;
    background:linear-gradient(135deg,#3b82f6,#8b5cf6);
    color:white;border-radius:12px;text-decoration:none;
    font-weight:700;font-size:1rem;margin-top:25px;transition:all 0.3s}
    .back-btn:hover{transform:translateY(-2px)}
    pre{background:#0f172a;padding:18px;border-radius:8px;
    color:#94a3b8;white-space:pre-wrap;font-size:0.88rem;
    line-height:1.7;margin-top:10px}
    @media(max-width:700px){
    .upload-grid,.steps{grid-template-columns:1fr}
    .summary-grid{grid-template-columns:repeat(2,1fr)}
    .nav{padding:15px 20px}.card{padding:20px}}
    """


# ==================== HOME PAGE ====================

@app.route("/")
def index():
    history_html = ""
    try:
        db = load_db()
        if db:
            history_html = """<div style="margin-top:30px;">
            <h3 style="color:#cbd5e1;margin-bottom:15px;">
            Previous Evaluations</h3>
            <div class="h-grid">"""
            for e in reversed(db):
                pct = e["percentage"]
                gc = (
                    "#22c55e" if pct >= 80 else
                    "#3b82f6" if pct >= 60 else
                    "#eab308" if pct >= 40 else
                    "#ef4444"
                )
                grade = e.get("grade", "N/A")
                desc = e.get("grade_description", "")
                history_html += f"""
                <a href="/results/{e['id']}" class="h-card">
                <div style="font-size:0.78rem;color:#64748b;">
                Q: {e.get('question_file','')}</div>
                <div style="font-size:0.78rem;color:#64748b;
                margin-bottom:10px;">
                A: {e.get('answer_file','')}</div>
                <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:5px;">
                <span style="color:{gc};font-weight:900;
                font-size:1.5rem;">{grade}</span>
                <span style="color:#3b82f6;font-weight:700;">
                {pct}%</span></div>
                <div style="color:#64748b;font-size:0.78rem;">
                {desc}</div>
                <div style="color:#475569;font-size:0.78rem;
                margin-top:8px;">
                {e['total_questions']} Qs | {e['total_marks']}/
                {e['max_marks']} marks</div></a>"""
            history_html += "</div></div>"
    except Exception:
        pass

    gemini_status = (
        f"Gemini: {EVALUATOR.gemini_model_name}"
        if EVALUATOR.gemini_model_name
        else "Gemini: Not connected"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI Paper Evaluator</title>
<style>{get_css()}</style></head><body>
<nav class="nav">
<h1>AI Paper Evaluator - Deep Learning</h1>
<span style="color:#94a3b8;font-size:0.82rem;">
RNN + Transformer + Gemini</span>
</nav>
<div class="container">
<div class="hero">
<h2>Smart Paper Evaluation System</h2>
<p>Upload Question Paper and Answer Sheet.
Get accurate grades A+ A B+ B C D F with detailed feedback.</p>
</div>

<div style="text-align:center;margin-bottom:20px;
font-size:0.82rem;color:#64748b;">
{gemini_status} |
PyTorch: {"ON" if HAS_TORCH else "OFF"} |
Transformer: {"ON" if HAS_ST else "OFF"}
</div>

<div class="steps">
<div class="step"><div class="step-num">1</div>
<h4>Upload Question PDF</h4>
<p>Exam question paper</p></div>
<div class="step"><div class="step-num">2</div>
<h4>Upload Answer PDF</h4>
<p>Student answer sheet</p></div>
<div class="step"><div class="step-num">3</div>
<h4>AI Grades Answers</h4>
<p>A+ to F with feedback</p></div>
</div>

<div class="card">
<form action="/upload" method="POST"
enctype="multipart/form-data" id="form">

<div class="upload-grid">
<div class="upload-box question-box">
<div class="file-tag q-tag">QUESTION PAPER</div>
<h3>Upload Question PDF</h3>
<p>PDF containing exam questions</p>
<input type="file" name="question_file"
accept=".pdf" required id="qFile">
<div class="file-sel" id="qName"></div>
</div>
<div class="upload-box answer-box">
<div class="file-tag a-tag">ANSWER SHEET</div>
<h3>Upload Answer PDF</h3>
<p>PDF containing student answers</p>
<input type="file" name="answer_file"
accept=".pdf" required id="aFile">
<div class="file-sel" id="aName"></div>
</div>
</div>

<div class="form-group">
<label>Maximum Marks Per Question</label>
<select name="max_marks">
<option value="5">5 Marks</option>
<option value="10" selected>10 Marks</option>
<option value="15">15 Marks</option>
<option value="20">20 Marks</option>
<option value="25">25 Marks</option>
<option value="50">50 Marks</option>
<option value="100">100 Marks</option>
</select>
</div>

<div class="info-box">
<strong>Grading System:</strong>
A+ (95%+) | A (80-94%) | B+ (70-79%) |
B (60-69%) | C+ (55-59%) | C (50-54%) |
D (40-49%) | E (33-39%) | F (below 33%)<br><br>
<strong>Format:</strong>
Questions PDF: Q1. question | Answers PDF: Q1. answer
</div>

<button type="submit" class="btn" id="btn">
Evaluate with AI Deep Learning
</button>
<div class="loading" id="loading">
<div class="spinner"></div>
<p>Running Deep Learning Evaluation...</p>
<p style="font-size:0.82rem;color:#64748b;margin-top:5px;">
Please wait 2-5 minutes</p>
</div>
</form>
</div>

{history_html}

<div class="card">
<h3 style="color:#cbd5e1;margin-bottom:15px;">PDF Format Guide</h3>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
<div>
<p style="color:#3b82f6;font-weight:600;margin-bottom:5px;">
Questions PDF</p>
<pre>Q1. What is Python?

Q2. What is OOP?
Explain its four pillars.

Q3. What is a database?

Q4. Explain loops in Python.

Q5. What is an API?</pre>
</div>
<div>
<p style="color:#8b5cf6;font-weight:600;margin-bottom:5px;">
Answers PDF</p>
<pre>Q1. Python is a high-level
programming language created
by Guido van Rossum in 1991.

Q2. OOP is Object Oriented
Programming. Four pillars are
encapsulation, inheritance,
polymorphism, abstraction.

Q3. A database is an organized
collection of structured data.

Q4. Loops repeat code. Python
has for loop and while loop.

Q5. API allows applications
to communicate with each other.</pre>
</div>
</div>
</div>
</div>

<script>
document.getElementById("qFile").addEventListener("change",
function(e){{
document.getElementById("qName").textContent=
e.target.files[0]?"Selected: "+e.target.files[0].name:""
}});
document.getElementById("aFile").addEventListener("change",
function(e){{
document.getElementById("aName").textContent=
e.target.files[0]?"Selected: "+e.target.files[0].name:""
}});
document.getElementById("form").addEventListener("submit",
function(){{
if(!document.getElementById("qFile").files.length||
!document.getElementById("aFile").files.length){{
alert("Please select both PDF files!");return false}}
document.getElementById("btn").disabled=true;
document.getElementById("btn").textContent=
"Evaluating... Please Wait";
document.getElementById("loading").style.display="block"
}});
</script>
</body></html>"""


# ==================== UPLOAD ====================

@app.route("/upload", methods=["POST"])
def upload():
    if ("question_file" not in request.files or
            "answer_file" not in request.files):
        return error_page("Both PDF files required.")

    qf = request.files["question_file"]
    af = request.files["answer_file"]

    if qf.filename == "" or af.filename == "":
        return error_page("Please select both PDFs.")
    if (not qf.filename.lower().endswith(".pdf") or
            not af.filename.lower().endswith(".pdf")):
        return error_page("Both files must be PDFs.")

    try:
        max_marks = int(request.form.get("max_marks", 10))
    except Exception:
        max_marks = 10

    q_fn = secure_filename(qf.filename)
    a_fn = secure_filename(af.filename)
    q_fp = os.path.join(UPLOAD_FOLDER, "Q_" + q_fn)
    a_fp = os.path.join(UPLOAD_FOLDER, "A_" + a_fn)
    qf.save(q_fp)
    af.save(a_fp)

    try:
        q_text = extract_text(q_fp)
        a_text = extract_text(a_fp)

        if not q_text or len(q_text) < 5:
            return error_page("Cannot read Question Paper PDF.")
        if not a_text or len(a_text) < 5:
            return error_page("Cannot read Answer Sheet PDF.")

        questions = extract_questions(q_text)
        answers = extract_answers(a_text)

        if not questions:
            return error_page(
                "No questions found. Use format: Q1. question"
            )
        if not answers:
            return error_page(
                "No answers found. Use format: Q1. answer"
            )

        qa_pairs = match_qa(questions, answers)
        results = evaluate_paper(qa_pairs, max_marks)
        eval_id = save_evaluation(q_fn, a_fn, results)
        return redirect(f"/results/{eval_id}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return error_page(str(e))


# ==================== RESULTS ====================

@app.route("/results/<int:eval_id>")
def results(eval_id):
    ev = get_evaluation(eval_id)
    if not ev:
        return error_page("Evaluation not found.")

    pct = ev["percentage"]
    grade = ev.get("grade", "N/A")
    grade_desc = ev.get("grade_description", "")

    # Grade color
    if grade in ["A+", "A"]:
        gc = "#22c55e"
    elif grade in ["B+", "B"]:
        gc = "#3b82f6"
    elif grade in ["C+", "C", "C-"]:
        gc = "#eab308"
    elif grade == "D":
        gc = "#f97316"
    else:
        gc = "#ef4444"

    q_html = ""
    for r in ev.get("results", []):
        marks = r.get("marks_obtained", 0)
        max_m = r.get("max_marks", 10)
        q_pct = round((marks / max_m) * 100) if max_m > 0 else 0

        # Question mark color
        if q_pct >= 80:
            mc = "#22c55e"
        elif q_pct >= 60:
            mc = "#3b82f6"
        elif q_pct >= 40:
            mc = "#eab308"
        else:
            mc = "#ef4444"

        # Badge class
        corr = r.get("correctness", "Average")
        badge_cls = {
            "Excellent": "badge-excellent",
            "Good": "badge-good",
            "Average": "badge-average",
            "Poor": "badge-poor"
        }.get(corr, "badge-average")

        # Strengths
        st_html = ""
        if r.get("strengths"):
            items = "".join(
                f"<li>+ {s}</li>" for s in r["strengths"]
            )
            st_html = f"""<div class="strength-box">
            <strong style="color:#22c55e;">Strengths:</strong>
            <ul>{items}</ul></div>"""

        # Improvements
        im_html = ""
        if r.get("improvements"):
            items = "".join(
                f"<li>- {i}</li>" for i in r["improvements"]
            )
            im_html = f"""<div class="improve-box">
            <strong style="color:#eab308;">
            Areas for Improvement:</strong>
            <ul>{items}</ul></div>"""

        # Score breakdown
        sb = r.get("score_breakdown", {})
        pills = ""
        if sb:
            def pill_class(val):
                if val == "N/A":
                    return "pill"
                try:
                    v = float(val)
                    if v >= 70:
                        return "pill good"
                    elif v >= 45:
                        return "pill avg"
                    else:
                        return "pill poor"
                except Exception:
                    return "pill"

            for key, val in sb.items():
                pills += (
                    f'<div class="{pill_class(val)}">'
                    f'{key.title()}:<span>{val}%</span></div>'
                )

        sb_html = ""
        if pills:
            sb_html = f"""<div class="score-box">
            <strong style="color:#c4b5fd;">
            Score Breakdown (AI Components):</strong>
            <div class="score-pills">{pills}</div></div>"""

        # Mini grade for question
        q_grade, _ = calculate_grade(q_pct)

        q_html += f"""
        <div class="q-card">
        <div class="q-head">
        <span class="q-num">Q{r.get("question_no","")}</span>
        <span style="font-weight:700;color:{mc};font-size:1rem;">
        {marks}/{max_m} marks</span>
        <span style="color:{mc};font-weight:600;font-size:0.9rem;">
        ({q_pct}%)</span>
        <span style="color:{mc};font-weight:700;font-size:0.9rem;
        background:rgba(0,0,0,0.2);padding:3px 10px;
        border-radius:6px;">Grade: {q_grade}</span>
        <span class="badge {badge_cls}">{corr}</span>
        </div>
        <div class="q-body">
        <div class="q-section"><strong>Question:</strong>
        <p>{r.get("question","")}</p></div>
        <div class="q-section"><strong>Student Answer:</strong>
        <p>{r.get("student_answer","")}</p></div>
        <div class="feedback-box">
        <strong style="color:#93c5fd;">AI Feedback:</strong>
        <p style="color:#94a3b8;margin-top:6px;line-height:1.7;">
        {r.get("feedback","")}</p>
        </div>
        {st_html}{im_html}{sb_html}
        </div></div>"""

    # Progress bar color
    prog_color = (
        "linear-gradient(90deg,#22c55e,#16a34a)" if pct >= 80 else
        "linear-gradient(90deg,#3b82f6,#2563eb)" if pct >= 60 else
        "linear-gradient(90deg,#eab308,#ca8a04)" if pct >= 40 else
        "linear-gradient(90deg,#ef4444,#dc2626)"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Results - AI Paper Evaluator</title>
<style>{get_css()}</style></head><body>
<nav class="nav">
<h1>AI Paper Evaluator</h1>
<a href="/">New Evaluation</a>
</nav>
<div class="container">
<div style="text-align:center;margin-bottom:30px;">
<h2 style="font-size:1.8rem;margin-bottom:8px;">
Evaluation Results</h2>
<p style="color:#64748b;font-size:0.82rem;">
Q: {ev.get('question_file','')} |
A: {ev.get('answer_file','')}</p>
<p style="color:#8b5cf6;font-size:0.78rem;margin-top:3px;">
Hybrid RNN + Transformer + Gemini AI</p>
</div>

<div class="card">
<div class="summary-grid">
<div class="stat"><span class="stat-label">Total Marks</span>
<span class="stat-val" style="color:#3b82f6;">
{ev['total_marks']}/{ev['max_marks']}</span></div>
<div class="stat"><span class="stat-label">Percentage</span>
<span class="stat-val" style="color:#3b82f6;">{pct}%</span></div>
<div class="stat"><span class="stat-label">Grade</span>
<span class="stat-val" style="color:{gc};">{grade}</span></div>
<div class="stat"><span class="stat-label">Questions</span>
<span class="stat-val" style="color:#3b82f6;">
{ev['total_questions']}</span></div>
</div>

<div class="prog-bg">
<div class="prog-fill" style="width:{pct}%;
background:{prog_color};"></div>
</div>

<div class="grade-box">
<div class="grade-letter" style="color:{gc};">{grade}</div>
<div style="color:{gc};font-weight:600;font-size:1.1rem;
margin-bottom:5px;">{pct}% Score</div>
<div class="grade-desc">{grade_desc}</div>
</div>

<div style="display:grid;grid-template-columns:repeat(3,1fr);
gap:10px;margin-top:15px;text-align:center;">
<div style="background:#0f172a;padding:12px;border-radius:8px;">
<div style="color:#64748b;font-size:0.78rem;">Passed Questions</div>
<div style="color:#22c55e;font-weight:700;font-size:1.1rem;">
{sum(1 for r in ev.get('results',[]) if r.get('percentage',0) >= 50)}
/{ev['total_questions']}</div>
</div>
<div style="background:#0f172a;padding:12px;border-radius:8px;">
<div style="color:#64748b;font-size:0.78rem;">Avg per Question</div>
<div style="color:#3b82f6;font-weight:700;font-size:1.1rem;">
{round(ev['total_marks']/ev['total_questions'],1) if ev['total_questions'] > 0 else 0}
/{ev['max_marks']//ev['total_questions'] if ev['total_questions'] > 0 else ev['max_marks']}</div>
</div>
<div style="background:#0f172a;padding:12px;border-radius:8px;">
<div style="color:#64748b;font-size:0.78rem;">Best Question</div>
<div style="color:#22c55e;font-weight:700;font-size:1.1rem;">
Q{max(ev.get('results',[{}]),key=lambda x:x.get('percentage',0)).get('question_no','N/A')}
({max((r.get('percentage',0) for r in ev.get('results',[])),default=0)}%)</div>
</div>
</div>
</div>

<h3 style="color:#cbd5e1;margin-bottom:20px;">
Question-wise Analysis</h3>
{q_html}
<a href="/" class="back-btn">Evaluate Another Paper</a>
</div></body></html>"""


# ==================== ERROR PAGE ====================

def error_page(msg):
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Error</title>
<style>{get_css()}</style></head><body>
<nav class="nav"><h1>AI Paper Evaluator</h1>
<a href="/">Home</a></nav>
<div class="container" style="display:flex;
justify-content:center;align-items:center;min-height:80vh;">
<div class="card" style="text-align:center;
max-width:550px;border-color:#ef4444;">
<h2 style="color:#ef4444;margin-bottom:15px;">Error</h2>
<p style="color:#94a3b8;margin-bottom:25px;
line-height:1.7;">{msg}</p>
<a href="/" class="back-btn">Go Back</a>
</div></div></body></html>"""


# ==================== RUN ====================

if __name__ == "__main__":
    print("=" * 55)
    print("AI Paper Evaluator - Deep Learning Edition")
    print(f"PyTorch:     {'YES' if HAS_TORCH else 'NO'}")
    print(f"Transformer: {'YES' if HAS_ST else 'NO'}")
    print(f"Gemini:      {EVALUATOR.gemini_model_name or 'NOT FOUND'}")
    print(f"API Key:     {'SET' if GEMINI_API_KEY else 'NOT SET'}")
    print("Open: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, port=5000, host="0.0.0.0")