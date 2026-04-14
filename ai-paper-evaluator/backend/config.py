import os
from dotenv import load_dotenv

# Load .env from project root (one level up from backend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

print(f"📄 Loading .env from: {ENV_PATH}")
print(f"🔑 API Key Found: {'✅ Yes' if os.getenv('GEMINI_API_KEY') else '❌ No'}")


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_EXTENSIONS = {"pdf"}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///evaluations.db")