import os

class Config:
    SECRET_KEY = "supersecretkey"

    SQLALCHEMY_DATABASE_URI = "postgresql://postgres:baba@localhost:5432/retail_db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    MODEL_FOLDER = os.path.join(os.getcwd(), "saved_models")