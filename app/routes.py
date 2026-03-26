import os
from datetime import datetime
from io import BytesIO

import pandas as pd
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app import db
from app.analytics import (
    analyze_uploaded_dataset,
    format_display_value,
    load_dataset_analysis,
    save_dataset_analysis,
)
from app.decorators import roles_required
from app.ml_model import predict_next_month, sync_model_with_sales_data
from app.models import Product, Sale, Store, User
from app.utils import (
    build_retail_analysis_payload,
    build_report_rows,
    build_report_summary,
    build_sales_chart_payload,
    build_sales_csv_file,
    build_sales_dataframe,
    build_sales_overview_cards,
)

main_routes = Blueprint("main_routes", __name__)

MAX_RETAIL_IMPORT_ROWS = 5000


def normalize_column_name(value):
    return str(value).strip().lower().replace("-", " ").replace("_", " ")


def column_match_score(column_name, aliases):
    normalized_column = normalize_column_name(column_name)
    column_tokens = set(normalized_column.split())
    best_score = 0

    for alias in aliases:
        normalized_alias = normalize_column_name(alias)
        alias_tokens = set(normalized_alias.split())

        if normalized_column == normalized_alias:
            best_score = max(best_score, 4)
        elif column_tokens and column_tokens == alias_tokens:
            best_score = max(best_score, 3)
        elif normalized_alias in normalized_column or normalized_column in normalized_alias:
            best_score = max(best_score, 2)
        elif alias_tokens and alias_tokens.issubset(column_tokens):
            best_score = max(best_score, 1)

    return best_score


def detect_columns(columns):
    mapping = {}
    used_columns = set()

    possible_names = {
        "store_name": ["store", "store_name", "shop", "branch", "store name", "shop name"],
        "city": ["city", "town", "location"],
        "state": ["state", "province", "region"],
        "product_name": ["product", "product_name", "item", "product name", "item name"],
        "category": ["category", "type", "class", "group", "segment"],
        "price": ["price", "unit_price", "cost", "unit price", "rate"],
        "quantity": ["quantity", "qty", "units", "count", "volume"],
        "revenue": ["revenue", "sales", "total", "amount", "value", "income"],
        "sale_date": ["date", "sale_date", "sale date", "transaction_date", "transaction date", "time", "datetime"],
    }

    for required_column, aliases in possible_names.items():
        ranked_matches = []
        for original_column in columns:
            if original_column in used_columns:
                continue

            score = column_match_score(original_column, aliases)
            if score > 0:
                ranked_matches.append((score, len(str(original_column)), original_column))

        if ranked_matches:
            ranked_matches.sort(reverse=True)
            best_match = ranked_matches[0][2]
            mapping[best_match] = required_column
            used_columns.add(best_match)

    essential_columns = ["product_name", "quantity", "revenue", "sale_date"]
    if not all(required in mapping.values() for required in essential_columns):
        return None

    return mapping


def build_empty_dashboard_payload():
    return {
        "dataset_name": "No dataset uploaded yet",
        "summary_cards": [
            {"label": "Records", "value": "0"},
            {"label": "Rows", "value": "0"},
            {"label": "Columns", "value": "0"},
            {"label": "Missing Cells", "value": "0"},
        ],
        "charts": {
            "trend": {"title": "Dataset Trend", "labels": [], "values": [], "dataset_label": ""},
            "breakdown": {"title": "Category Breakdown", "labels": [], "values": [], "dataset_label": ""},
            "composition": {"title": "Composition", "labels": [], "values": [], "dataset_label": ""},
            "distribution": {"title": "Distribution", "labels": [], "values": [], "dataset_label": ""},
        },
        "insights": {
            "analysis_note": "Upload a CSV or Excel file to let the dashboard detect the most useful columns automatically.",
            "date_column": "Not detected",
            "metric_column": "Not detected",
            "category_column": "Not detected",
            "top_segment": "Not available",
        },
    }


def build_dashboard_payload():
    cached_payload = load_dataset_analysis()
    retail_dataset_name = (
        cached_payload.get("dataset_name", "Retail Sales Records")
        if cached_payload
        else "Retail Sales Records"
    )
    retail_payload = build_retail_analysis_payload(retail_dataset_name)
    if retail_payload:
        return retail_payload

    payload = cached_payload
    if payload:
        payload.setdefault("summary_cards", [])
        payload.setdefault("charts", {})
        payload.setdefault("insights", {})
        return payload

    sales_df = build_sales_dataframe()
    if sales_df.empty:
        return build_empty_dashboard_payload()

    try:
        return analyze_uploaded_dataset(sales_df, "Retail Sales Records")
    except Exception:
        return build_empty_dashboard_payload()


def build_detected_fields(payload):
    insights = payload.get("insights", {})
    return [
        {"label": "Date Column", "value": insights.get("date_column", "Not detected")},
        {"label": "Metric Column", "value": insights.get("metric_column", "Not detected")},
        {"label": "Grouping Column", "value": insights.get("category_column", "Not detected")},
        {"label": insights.get("top_segment_label", "Top Segment"), "value": insights.get("top_segment", "Not available")},
    ]


def clean_text(value, default_value):
    if pd.isna(value):
        return default_value

    cleaned = str(value).strip()
    return cleaned or default_value


def load_uploaded_dataframe(file_path, filename):
    if filename.lower().endswith(".csv"):
        read_attempts = (
            {"encoding": "utf-8", "low_memory": False},
            {"encoding": "utf-8-sig", "low_memory": False},
            {"encoding": "latin1", "low_memory": False},
            {"encoding": "cp1252", "low_memory": False},
        )

        last_error = None
        for options in read_attempts:
            try:
                return pd.read_csv(file_path, **options)
            except UnicodeDecodeError as exc:
                last_error = exc

        if last_error:
            raise last_error

    return pd.read_excel(file_path)


def import_retail_rows(dataframe):
    working_df = dataframe.copy()

    if "store_name" not in working_df.columns:
        working_df["store_name"] = "Unknown Store"
    if "city" not in working_df.columns:
        working_df["city"] = "Unknown"
    if "state" not in working_df.columns:
        working_df["state"] = "Unknown"
    if "category" not in working_df.columns:
        working_df["category"] = "General"

    working_df["quantity"] = pd.to_numeric(working_df["quantity"], errors="coerce")
    working_df["revenue"] = pd.to_numeric(working_df["revenue"], errors="coerce")

    if "price" in working_df.columns:
        working_df["price"] = pd.to_numeric(working_df["price"], errors="coerce")
    else:
        working_df["price"] = None

    valid_quantity = working_df["quantity"].replace(0, pd.NA)
    inferred_price = working_df["revenue"] / valid_quantity
    working_df["price"] = working_df["price"].fillna(inferred_price).fillna(0)
    working_df["sale_date"] = pd.to_datetime(working_df["sale_date"], errors="coerce")

    required_columns = ["store_name", "product_name", "quantity", "revenue", "sale_date"]
    working_df.dropna(subset=required_columns, inplace=True)
    if working_df.empty:
        return 0

    store_cache = {store.name: store for store in Store.query.all()}
    product_cache = {product.name: product for product in Product.query.all()}

    imported_rows = 0
    for _, row in working_df.iterrows():
        store_name = clean_text(row["store_name"], "Unknown Store")
        product_name = clean_text(row["product_name"], "Unknown Product")
        if not store_name or not product_name:
            continue

        store = store_cache.get(store_name)
        if not store:
            store = Store(
                name=store_name,
                city=clean_text(row["city"], "Unknown"),
                state=clean_text(row["state"], "Unknown"),
            )
            db.session.add(store)
            db.session.flush()
            store_cache[store_name] = store

        product = product_cache.get(product_name)
        if not product:
            product = Product(
                name=product_name,
                category=clean_text(row["category"], "General"),
                price=float(row["price"]),
            )
            db.session.add(product)
            db.session.flush()
            product_cache[product_name] = product

        sale = Sale(
            store_id=store.id,
            product_id=product.id,
            quantity=int(float(row["quantity"])),
            revenue=float(row["revenue"]),
            sale_date=row["sale_date"],
        )

        db.session.add(sale)
        imported_rows += 1

    db.session.commit()
    return imported_rows


@main_routes.route("/")
@login_required
def dashboard():
    payload = build_dashboard_payload()
    model_info = sync_model_with_sales_data()
    sales_overview_cards = build_sales_overview_cards()
    next_month_prediction = predict_next_month()

    default_chart = {"title": "", "labels": [], "values": [], "dataset_label": ""}

    if next_month_prediction is not None:
        sales_overview_cards.append(
            {
                "label": "Next Month Forecast",
                "value": format_display_value(next_month_prediction),
            }
        )

    return render_template(
        "dashboard.html",
        user=current_user,
        active_dataset_name=payload.get("dataset_name", "Uploaded Dataset"),
        summary_cards=payload.get("summary_cards", []),
        trend_chart=payload.get("charts", {}).get("trend", default_chart),
        bar_chart=payload.get("charts", {}).get("breakdown", default_chart),
        pie_chart=payload.get("charts", {}).get("composition", default_chart),
        distribution_chart=payload.get("charts", {}).get("distribution", default_chart),
        analysis_note=payload.get("insights", {}).get("analysis_note", ""),
        detected_fields=build_detected_fields(payload),
        model_info=model_info,
        model_last_trained=model_info.get("trained_at_label") if model_info else None,
        sales_overview_cards=sales_overview_cards,
        sales_charts=build_sales_chart_payload(),
        dashboard_refresh_ms=current_app.config["DASHBOARD_REFRESH_INTERVAL_MS"],
    )


@main_routes.route("/admin/users", methods=["GET", "POST"])
@roles_required("admin")
def manage_users():
    if request.method == "POST":
        action = (request.form.get("action") or "update_role").strip().lower()

        if action == "create_user":
            username = (request.form.get("username") or "").strip()
            email = (request.form.get("email") or "").strip().lower()
            password = request.form.get("password") or ""
            selected_role = (request.form.get("role") or "").strip().lower()

            if not username or not email or not password:
                flash("Username, email, and password are required to create a user.", "danger")
                return redirect(url_for("main_routes.manage_users"))

            if selected_role not in User.ROLE_CHOICES:
                flash("Invalid role selected.", "danger")
                return redirect(url_for("main_routes.manage_users"))

            if User.query.filter_by(username=username).first():
                flash("That username already exists.", "danger")
                return redirect(url_for("main_routes.manage_users"))

            if User.query.filter_by(email=email).first():
                flash("That email already exists.", "danger")
                return redirect(url_for("main_routes.manage_users"))

            new_user = User(username=username, email=email, role=selected_role)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash(f"Created {selected_role} account for {username}.", "success")
            return redirect(url_for("main_routes.manage_users"))

        user_id = request.form.get("user_id", type=int)
        selected_role = (request.form.get("role") or "").strip().lower()

        if selected_role not in User.ROLE_CHOICES:
            flash("Invalid role selected.", "danger")
            return redirect(url_for("main_routes.manage_users"))

        user = db.session.get(User, user_id)
        if user is None:
            flash("User not found.", "danger")
            return redirect(url_for("main_routes.manage_users"))

        if user.id == current_user.id and selected_role != User.ROLE_ADMIN:
            flash("You cannot remove your own admin access from this page.", "warning")
            return redirect(url_for("main_routes.manage_users"))

        user.role = selected_role
        db.session.commit()
        flash(f"Updated role for {user.username} to {selected_role}.", "success")
        return redirect(url_for("main_routes.manage_users"))

    users = User.query.order_by(User.username.asc()).all()
    return render_template(
        "users.html",
        users=users,
        available_roles=User.ROLE_CHOICES,
    )


@main_routes.route("/api/dashboard/sales-summary")
@login_required
def dashboard_sales_summary():
    return jsonify(build_sales_chart_payload())


@main_routes.route("/upload", methods=["GET", "POST"])
@roles_required("admin")
def upload_data():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("No file selected", "danger")
            return redirect(request.url)

        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        try:
            dataframe = load_uploaded_dataframe(file_path, file.filename)
            analysis = analyze_uploaded_dataset(dataframe, file.filename)
            column_mapping = detect_columns(dataframe.columns.tolist())
            analysis["retail_compatible"] = False
            save_dataset_analysis(analysis)
        except Exception as exc:
            flash(f"Error while reading the file: {str(exc)}", "danger")
            return redirect(request.url)

        if not column_mapping:
            flash("Dataset uploaded successfully. The dashboard is now analyzing it dynamically.", "success")
            flash("Retail forecasting was skipped because the file does not look like sales data.", "info")
            return redirect(url_for("main_routes.dashboard"))

        if len(dataframe) > MAX_RETAIL_IMPORT_ROWS:
            flash("Large dataset uploaded successfully. The dashboard is analyzing it directly.", "success")
            flash(
                (
                    f"Retail database import was skipped because the file has {len(dataframe):,} rows. "
                    f"The app keeps dynamic analysis active for large files above {MAX_RETAIL_IMPORT_ROWS:,} rows."
                ),
                "info",
            )
            return redirect(url_for("main_routes.dashboard"))

        retail_df = dataframe.rename(columns=column_mapping).copy()

        try:
            imported_rows = import_retail_rows(retail_df)
            if imported_rows > 0:
                analysis["retail_compatible"] = True
                save_dataset_analysis(analysis)
                model_info = sync_model_with_sales_data(force=True)

                flash("Dataset uploaded successfully. Dynamic analysis and retail forecasting are ready.", "success")
                if model_info:
                    flash(
                        (
                            f"Retail model retrained automatically. "
                            f"Best model: {model_info['best_model_name']} (R2 {model_info['best_accuracy']})."
                        ),
                        "success",
                    )
            else:
                flash(
                    "Dataset uploaded successfully. Dynamic analysis is ready, but no valid sales rows were available for forecasting.",
                    "warning",
                )
        except Exception as exc:
            db.session.rollback()
            flash("Dataset uploaded successfully and generic analysis is ready.", "success")
            flash(f"Retail-specific import was skipped due to: {str(exc)}", "warning")

        return redirect(url_for("main_routes.dashboard"))

    return render_template("upload.html")


@main_routes.route("/train-model", methods=["GET", "POST"])
@roles_required("admin")
def train_model_route():
    model_info = sync_model_with_sales_data(force=True)
    if model_info:
        flash(
            (
                f"Model retrained successfully. "
                f"Best model: {model_info['best_model_name']} (R2 {model_info['best_accuracy']})."
            ),
            "success",
        )
    else:
        flash("Not enough sales data to train the retail forecast model.", "danger")

    return redirect(url_for("main_routes.dashboard"))


@main_routes.route("/reports")
@roles_required("admin", "manager")
def reports():
    return render_template(
        "reports.html",
        report_summary=build_report_summary(),
        report_rows=build_report_rows(limit=50),
    )


@main_routes.route("/reports/export/csv")
@roles_required("admin", "manager")
def export_sales_csv():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        build_sales_csv_file(),
        as_attachment=True,
        download_name=f"sales_report_{timestamp}.csv",
        mimetype="text/csv",
    )


@main_routes.route("/download-report")
@roles_required("admin", "manager")
def download_report():
    report_summary = build_report_summary()

    buffer = BytesIO()
    document = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Retail Intelligence Dashboard Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.4 * inch))
    elements.append(Paragraph(f"Total sales records: {report_summary['total_sales_display']}", styles["BodyText"]))
    elements.append(Paragraph(f"Total revenue: {report_summary['total_revenue_display']}", styles["BodyText"]))
    elements.append(Paragraph(f"Units sold: {report_summary['total_units_display']}", styles["BodyText"]))
    elements.append(Paragraph(f"Average sale value: {report_summary['average_revenue_display']}", styles["BodyText"]))

    document.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="sales_prediction_report.pdf",
        mimetype="application/pdf",
    )
