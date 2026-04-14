import google.generativeai as genai
import json
import re


class AIEvaluator:
    """Evaluate answers using Google Gemini AI"""

    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def evaluate_single_answer(self, question, answer, max_marks=10):
        """Evaluate a single question-answer pair"""

        prompt = f"""
You are an expert exam evaluator. Evaluate the following answer strictly and fairly.

**Question:** {question}

**Student's Answer:** {answer}

**Maximum Marks:** {max_marks}

Evaluate based on:
1. Correctness - Is the answer factually correct?
2. Completeness - Does it cover all important points?
3. Clarity - Is it well-explained and clear?
4. Relevance - Does it directly answer the question?

Respond ONLY in this exact JSON format:
{{
    "marks_obtained": <number>,
    "max_marks": {max_marks},
    "percentage": <number>,
    "feedback": "<detailed feedback>",
    "strengths": ["<strength1>", "<strength2>"],
    "improvements": ["<improvement1>", "<improvement2>"],
    "correctness": "<Excellent/Good/Average/Poor>",
    "completeness": "<Excellent/Good/Average/Poor>"
}}
"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Clean the response to extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {
                    "marks_obtained": 0,
                    "max_marks": max_marks,
                    "percentage": 0,
                    "feedback": "Could not evaluate this answer.",
                    "strengths": [],
                    "improvements": ["Answer could not be parsed"],
                    "correctness": "N/A",
                    "completeness": "N/A"
                }
        except Exception as e:
            return {
                "marks_obtained": 0,
                "max_marks": max_marks,
                "percentage": 0,
                "feedback": f"Evaluation error: {str(e)}",
                "strengths": [],
                "improvements": [],
                "correctness": "Error",
                "completeness": "Error"
            }

    def evaluate_full_paper(self, qa_pairs, max_marks_per_question=10):
        """Evaluate all questions in a paper"""

        results = []
        total_marks = 0
        total_max = 0

        for qa in qa_pairs:
            result = self.evaluate_single_answer(
                question=qa["question"],
                answer=qa["answer"],
                max_marks=max_marks_per_question
            )
            result["question_no"] = qa["question_no"]
            result["question"] = qa["question"]
            result["student_answer"] = qa["answer"]
            results.append(result)

            total_marks += result.get("marks_obtained", 0)
            total_max += max_marks_per_question

        summary = {
            "total_marks": total_marks,
            "total_max_marks": total_max,
            "percentage": round((total_marks / total_max) * 100, 2) if total_max > 0 else 0,
            "total_questions": len(qa_pairs),
            "grade": self._calculate_grade(total_marks, total_max),
            "results": results
        }

        return summary

    @staticmethod
    def _calculate_grade(marks, total):
        """Calculate grade based on percentage"""
        if total == 0:
            return "N/A"
        percentage = (marks / total) * 100
        if percentage >= 90:
            return "A+"
        elif percentage >= 80:
            return "A"
        elif percentage >= 70:
            return "B+"
        elif percentage >= 60:
            return "B"
        elif percentage >= 50:
            return "C"
        elif percentage >= 40:
            return "D"
        else:
            return "F"