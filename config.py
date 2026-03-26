import os


class Config:
    SECRET_KEY = "supersecretkey"
    SQLALCHEMY_DATABASE_URI = "sqlite:///retail.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    MODEL_FOLDER = os.path.join(os.getcwd(), "saved_models")
    DASHBOARD_REFRESH_INTERVAL_MS = 30000
    DEFAULT_USER_ROLE = "manager"
