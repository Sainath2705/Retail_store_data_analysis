import os

from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

from config import Config

db = SQLAlchemy()
login_manager = LoginManager()

login_manager.login_view = "auth_routes.login"
login_manager.login_message_category = "warning"


def _normalize_existing_roles():
    from app.models import User

    legacy_users = User.query.filter(User.role.in_(["user", "staff"])).all()
    if not legacy_users:
        return

    for user in legacy_users:
        user.role = User.ROLE_MANAGER

    db.session.commit()


def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(Config)

    if isinstance(config_object, dict):
        app.config.update(config_object)
    else:
        app.config.from_object(config_object)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from app.models import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from app.auth import auth_routes
    from app.routes import main_routes

    app.register_blueprint(auth_routes)
    app.register_blueprint(main_routes)

    with app.app_context():
        db.create_all()
        _normalize_existing_roles()

    return app
