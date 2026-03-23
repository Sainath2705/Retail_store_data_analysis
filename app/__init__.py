from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config

db = SQLAlchemy()
login_manager = LoginManager()

# IMPORTANT
login_manager.login_view = "auth_routes.login"


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)

    from app.models import User

    # 🔐 THIS IS THE FIX
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    with app.app_context():
        db.create_all()

    from app.auth import auth_routes
    from app.routes import main_routes

    app.register_blueprint(auth_routes)
    app.register_blueprint(main_routes)

    return app