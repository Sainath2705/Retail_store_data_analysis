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
from app.ai_insights import ask_question, generate_auto_summary
from app.ml_model import (
    get_cached_uploaded_prediction,
    predict_next_month,
    sync_model_with_sales_data,
)
from app.models import Product, Sale, Store, User
from app.utils import (
    build_retail_analysis_payload,
    build_report_rows,
    build_report_summary,
    build_sales_chart_payload,
    build_sales_csv_file,
    build_sales_dataframe,
    build_sales_overview_cards,
    build_sales_performance_payload,
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
        "analysis_source": "uploaded_dataset",
        "dataset_name": "No dataset uploaded yet",
        "summary_cards": [
            {"label": "Records", "value": "0"},
            {"label": "Rows", "value": "0"},
            {"label": "Columns", "value": "0"},
            {"label": "Missing Cells", "value": "0"},
        ],
        "business_metrics": {
            "title": "Sales Performance",
            "subtitle": "Upload a retail dataset to calculate sales performance metrics.",
            "note": "Profit metrics such as Gross Profit, EBITDA, and Net Profit need cost and expense data.",
            "cards": [],
            "charts": {},
        },
        "insight_cards": [],
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


def _chart_has_data(chart):
    return bool(chart and chart.get("labels") and chart.get("values"))


def _live_chart_from_uploaded_chart(chart, fallback_title, fallback_label):
    chart = chart or {}
    return {
        "title": chart.get("title") or fallback_title,
        "labels": chart.get("labels") or [],
        "values": chart.get("values") or [],
        "dataset_label": chart.get("dataset_label") or fallback_label,
        "description": chart.get("description") or "",
    }


def build_live_sales_chart_payload(dataframe=None, user_id=None, dashboard_payload=None):
    sales_payload = build_sales_chart_payload(dataframe=dataframe, user_id=user_id)
    if any(_chart_has_data(chart) for chart in sales_payload.values()):
        return sales_payload

    dashboard_payload = dashboard_payload or load_dataset_analysis()
    uploaded_charts = (dashboard_payload or {}).get("charts", {})
    fallback_payload = {
        "daily": _live_chart_from_uploaded_chart(
            uploaded_charts.get("trend"),
            "Uploaded Dataset Trend",
            "Trend",
        ),
        "weekly": _live_chart_from_uploaded_chart(
            uploaded_charts.get("breakdown"),
            "Uploaded Dataset Breakdown",
            "Breakdown",
        ),
        "monthly": _live_chart_from_uploaded_chart(
            uploaded_charts.get("composition") or uploaded_charts.get("distribution"),
            "Uploaded Dataset Share",
            "Share",
        ),
    }

    if any(_chart_has_data(chart) for chart in fallback_payload.values()):
        return fallback_payload

    return sales_payload


def build_dashboard_payload(dataframe=None, user_id=None):
    cached_payload = load_dataset_analysis()

    payload = cached_payload
    if payload:
        payload.setdefault("summary_cards", [])
        payload.setdefault("insight_cards", [])
        payload.setdefault("charts", {})
        payload.setdefault("insights", {})
        payload.setdefault("analysis_source", "uploaded_dataset")
        if not payload.get("retail_compatible"):
            return payload

    if dataframe is None:
        sales_df = build_sales_dataframe(user_id=user_id)
    else:
        sales_df = dataframe

    retail_dataset_name = (
        cached_payload.get("dataset_name", "Retail Sales Records")
        if cached_payload
        else "Retail Sales Records"
    )
    retail_payload = build_retail_analysis_payload(retail_dataset_name, dataframe=sales_df, user_id=user_id)
    if retail_payload:
        return retail_payload

    if sales_df.empty:
        return build_empty_dashboard_payload()

    try:
        return analyze_uploaded_dataset(sales_df, "Retail Sales Records")
    except Exception:
        return build_empty_dashboard_payload()


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


def import_retail_rows(dataframe, user_id=None):
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
            user_id=user_id,
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


def _remove_files_in_folder(folder_path):
    removed_files = 0
    if not folder_path or not os.path.isdir(folder_path):
        return removed_files

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            removed_files += 1

    return removed_files


def clear_dashboard_data():
    removed_upload_files = _remove_files_in_folder(current_app.config.get("UPLOAD_FOLDER"))
    removed_model_files = _remove_files_in_folder(current_app.config.get("MODEL_FOLDER"))

    analysis_cache_path = os.path.join(current_app.instance_path, "dataset_analysis.json")
    removed_cache_files = 0
    if os.path.exists(analysis_cache_path):
        os.remove(analysis_cache_path)
        removed_cache_files = 1

    deleted_sales = db.session.query(Sale).delete(synchronize_session=False)
    db.session.query(Product).delete(synchronize_session=False)
    db.session.query(Store).delete(synchronize_session=False)
    db.session.commit()

    return {
        "removed_upload_files": removed_upload_files,
        "removed_model_files": removed_model_files,
        "removed_cache_files": removed_cache_files,
        "deleted_sales": deleted_sales,
    }


@main_routes.route("/")
@login_required
def dashboard():
    dataframe = build_sales_dataframe(user_id=current_user.id)
    payload = build_dashboard_payload(dataframe=dataframe, user_id=current_user.id)
    model_info = sync_model_with_sales_data() if not dataframe.empty else None
    sales_overview_cards = build_sales_overview_cards(dataframe=dataframe, user_id=current_user.id)
    next_month_prediction = predict_next_month() if not dataframe.empty else None
    uploaded_prediction = None

    if dataframe.empty and payload.get("summary_cards"):
        sales_overview_cards = payload.get("summary_cards", [])
        uploaded_prediction = get_cached_uploaded_prediction()

    default_chart = {"title": "", "labels": [], "values": [], "dataset_label": "", "description": ""}

    if next_month_prediction is not None:
        sales_overview_cards.append(
            {
                "label": "Next Month Forecast",
                "value": format_display_value(next_month_prediction),
            }
        )
    elif uploaded_prediction is not None:
        sales_overview_cards.append(
            {
                "label": f"Next Month {uploaded_prediction['metric_column']} Forecast",
                "value": format_display_value(uploaded_prediction["prediction"]),
            }
        )

    business_metrics = payload.get("business_metrics")
    if not business_metrics and not dataframe.empty:
        business_metrics = build_sales_performance_payload(dataframe=dataframe, user_id=current_user.id)
    if not business_metrics:
        business_metrics = {
            "title": "Sales Performance",
            "subtitle": "Upload a retail dataset to calculate sales performance metrics.",
            "note": "Profit metrics such as Gross Profit, EBITDA, and Net Profit need cost and expense data.",
            "cards": [],
            "charts": {},
        }

    return render_template(
        "dashboard.html",
        user=current_user,
        active_dataset_name=payload.get("dataset_name", "Uploaded Dataset"),
        summary_cards=payload.get("summary_cards", []),
        insight_cards=payload.get("insight_cards", []),
        trend_chart=payload.get("charts", {}).get("trend", default_chart),
        bar_chart=payload.get("charts", {}).get("breakdown", default_chart),
        pie_chart=payload.get("charts", {}).get("composition", default_chart),
        distribution_chart=payload.get("charts", {}).get("distribution", default_chart),
        business_metrics=business_metrics,
        analysis_note=payload.get("insights", {}).get("analysis_note", ""),
        analysis_source=payload.get("analysis_source", "uploaded_dataset"),
        model_info=model_info,
        model_last_trained=model_info.get("trained_at_label") if model_info else None,
        sales_overview_cards=sales_overview_cards,
        sales_charts=build_live_sales_chart_payload(
            dataframe=dataframe,
            user_id=current_user.id,
            dashboard_payload=payload,
        ),
        live_charts_are_sales=not dataframe.empty,
        uploaded_prediction=uploaded_prediction,
        dashboard_refresh_ms=current_app.config["DASHBOARD_REFRESH_INTERVAL_MS"],
    )


@main_routes.route("/delete-data", methods=["POST"])
@roles_required("admin")
def delete_data():
    try:
        result = clear_dashboard_data()
    except Exception as exc:
        db.session.rollback()
        flash(f"Could not clear dashboard data: {str(exc)}", "danger")
        return redirect(url_for("main_routes.dashboard"))

    flash(
        (
            "Dashboard data cleared successfully. "
            f"Removed {result['deleted_sales']:,} sales rows, "
            f"{result['removed_upload_files']} uploaded files, "
            f"{result['removed_model_files']} model files, and "
            f"{result['removed_cache_files']} cached analysis file."
        ),
        "success",
    )
    return redirect(url_for("main_routes.dashboard"))


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
    return jsonify(build_live_sales_chart_payload(user_id=current_user.id))


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
            imported_rows = import_retail_rows(retail_df, user_id=current_user.id)
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
        report_summary=build_report_summary(user_id=current_user.id),
        report_rows=build_report_rows(limit=50, user_id=current_user.id),
    )


@main_routes.route("/reports/export/csv")
@roles_required("admin", "manager")
def export_sales_csv():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        build_sales_csv_file(user_id=current_user.id),
        as_attachment=True,
        download_name=f"sales_report_{timestamp}.csv",
        mimetype="text/csv",
    )


@main_routes.route("/download-report")
@roles_required("admin", "manager")
def download_report():
    report_summary = build_report_summary(user_id=current_user.id)

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


@main_routes.route("/api/ai/summary")
@login_required
def ai_summary():
    summary = generate_auto_summary(user_id=current_user.id)
    if summary is None:
        return jsonify({"error": "AI summary unavailable. Check API key or upload data first."}), 503
    return jsonify({"summary": summary})


@main_routes.route("/api/ai/ask", methods=["POST"])
@login_required
def ai_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please provide a question."}), 400

    result = ask_question(question, user_id=current_user.id)
    if "error" in result:
        return jsonify(result), 503
    return jsonify(result)
