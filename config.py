import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "fallback-dev-key-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///retail.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    MODEL_FOLDER = os.path.join(os.getcwd(), "saved_models")
    DASHBOARD_REFRESH_INTERVAL_MS = 30000
    DEFAULT_USER_ROLE = "manager"
