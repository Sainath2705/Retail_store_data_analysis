from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from app import db


class User(db.Model, UserMixin):
    ROLE_ADMIN = "admin"
    ROLE_MANAGER = "manager"
    ROLE_CHOICES = (ROLE_ADMIN, ROLE_MANAGER)

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default=ROLE_MANAGER, nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def normalized_role(self):
        value = (self.role or self.ROLE_MANAGER).strip().lower()
        if value not in {self.ROLE_ADMIN, self.ROLE_MANAGER}:
            return self.ROLE_MANAGER
        return value

    def has_role(self, *roles):
        allowed_roles = {str(role).strip().lower() for role in roles}
        return self.normalized_role in allowed_roles

    def is_admin(self):
        return self.has_role(self.ROLE_ADMIN)

    def is_manager(self):
        return self.has_role(self.ROLE_MANAGER)


class Store(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    category = db.Column(db.String(100))
    price = db.Column(db.Float)


class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    store_id = db.Column(db.Integer, db.ForeignKey("store.id"))
    product_id = db.Column(db.Integer, db.ForeignKey("product.id"))
    quantity = db.Column(db.Integer)
    revenue = db.Column(db.Float)
    sale_date = db.Column(db.DateTime, default=datetime.utcnow)

    store = db.relationship("Store", backref=db.backref("sales", lazy=True))
    product = db.relationship("Product", backref=db.backref("sales", lazy=True))
