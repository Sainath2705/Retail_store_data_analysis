from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app import db
from app.models import User

auth_routes = Blueprint("auth_routes", __name__)


def ensure_admin_exists(fallback_user=None):
    users = User.query.order_by(User.id.asc()).all()
    if any(user.normalized_role == User.ROLE_ADMIN for user in users):
        return False

    selected_user = fallback_user or (users[0] if users else None)
    if selected_user is None:
        return False

    selected_user.role = User.ROLE_ADMIN
    db.session.commit()
    return True


@auth_routes.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main_routes.dashboard"))

    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "danger")
            return redirect(url_for("auth_routes.register"))

        if User.query.filter_by(email=email).first():
            flash("Email already exists!", "danger")
            return redirect(url_for("auth_routes.register"))

        assigned_role = User.ROLE_ADMIN if User.query.count() == 0 else User.ROLE_MANAGER
        user = User(username=username, email=email, role=assigned_role)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        if assigned_role == User.ROLE_ADMIN:
            flash("Account created successfully. The first user was assigned as admin.", "success")
        else:
            flash("Account created successfully! Please login.", "success")

        return redirect(url_for("auth_routes.login"))

    return render_template("register.html")


@auth_routes.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main_routes.dashboard"))

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            admin_bootstrapped = ensure_admin_exists(fallback_user=user)
            login_user(user)
            if admin_bootstrapped:
                flash("No admin account was found, so your account was promoted to admin.", "warning")
            flash("Welcome back!", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("main_routes.dashboard"))

        flash("Invalid credentials", "danger")

    return render_template("login.html")


@auth_routes.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully", "info")
    return redirect(url_for("auth_routes.login"))
