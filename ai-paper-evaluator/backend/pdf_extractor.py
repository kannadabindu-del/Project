import pdfplumber
import PyPDF2
import re


class PDFExtractor:
    """Extract text from PDF files"""

    @staticmethod
    def extract_with_pdfplumber(pdf_path):
        """Primary extraction method using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"pdfplumber error: {e}")
        return text

    @staticmethod
    def extract_with_pypdf2(pdf_path):
        """Fallback extraction method using PyPDF2"""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"PyPDF2 error: {e}")
        return text

    @staticmethod
    def extract_text(pdf_path):
        """Extract text using best available method"""
        text = PDFExtractor.extract_with_pdfplumber(pdf_path)
        if not text.strip():
            text = PDFExtractor.extract_with_pypdf2(pdf_path)
        return text.strip()

    @staticmethod
    def extract_questions_and_answers(text):
        """Parse questions and answers from extracted text"""
        qa_pairs = []

        # Pattern 1: Q1. Question\nA1. Answer
        pattern1 = re.findall(
            r'[Qq](?:uestion)?\s*(\d+)[.):]\s*(.*?)\n\s*[Aa](?:nswer)?\s*\1?[.):]\s*(.*?)(?=\n\s*[Qq](?:uestion)?\s*\d+|$)',
            text, re.DOTALL
        )

        if pattern1:
            for match in pattern1:
                qa_pairs.append({
                    "question_no": int(match[0]),
                    "question": match[1].strip(),
                    "answer": match[2].strip()
                })
            return qa_pairs

        # Pattern 2: Numbered questions with answers below
        pattern2 = re.findall(
            r'(\d+)[.):]\s*(.*?)\n\s*(?:Answer|Ans|A)[.):s]*\s*(.*?)(?=\n\s*\d+[.):]|$)',
            text, re.DOTALL
        )

        if pattern2:
            for match in pattern2:
                qa_pairs.append({
                    "question_no": int(match[0]),
                    "question": match[1].strip(),
                    "answer": match[2].strip()
                })
            return qa_pairs

        # Pattern 3: Simple split by blank lines
        sections = text.split("\n\n")
        q_num = 1
        for i in range(0, len(sections) - 1, 2):
            qa_pairs.append({
                "question_no": q_num,
                "question": sections[i].strip(),
                "answer": sections[i + 1].strip() if i + 1 < len(sections) else ""
            })
            q_num += 1

        return qa_pairs