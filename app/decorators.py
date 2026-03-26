from functools import wraps

from flask import flash, jsonify, redirect, request, url_for
from flask_login import current_user, login_required


def roles_required(*roles):
    def decorator(view_function):
        @wraps(view_function)
        @login_required
        def wrapped_view(*args, **kwargs):
            if current_user.has_role(*roles):
                return view_function(*args, **kwargs)

            if request.path.startswith("/api/"):
                return jsonify({"error": "Forbidden"}), 403

            flash("You do not have permission to access that page.", "danger")
            return redirect(url_for("main_routes.dashboard"))

        return wrapped_view

    return decorator
