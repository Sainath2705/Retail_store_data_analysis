from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)  # default template folder = app/templates

    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth_routes.login"

    from app.models import User
    from app.routes import main_routes
    from app.auth import auth_routes

    app.register_blueprint(main_routes)
    app.register_blueprint(auth_routes)

    with app.app_context():
        db.create_all()

    return app