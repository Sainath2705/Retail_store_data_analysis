import os
from datetime import datetime
from io import BytesIO

import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, send_file, url_for
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
from app.ml_model import predict_next_month, train_models
from app.models import Product, Sale, Store
from config import Config

main_routes = Blueprint("main_routes", __name__)

MODEL_INFO_CACHE = None
MODEL_LAST_TRAINED = None
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
    """
    Auto-detect and map columns to the retail schema when possible.
    Returns a dict {original_col: required_col} or None if the file is not sales-shaped.
    """
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


def build_sales_dataframe():
    sales = Sale.query.all()

    if not sales:
        return None

    product_map = {
        product.id: product
        for product in Product.query.all()
    }
    store_map = {
        store.id: store
        for store in Store.query.all()
    }

    rows = []
    for sale in sales:
        product = product_map.get(sale.product_id)
        store = store_map.get(sale.store_id)

        rows.append(
            {
                "store_name": store.name if store else "Unknown Store",
                "city": store.city if store else "Unknown",
                "state": store.state if store else "Unknown",
                "product_name": product.name if product else "Unknown Product",
                "category": product.category if product else "General",
                "price": product.price if product else 0,
                "quantity": sale.quantity,
                "revenue": sale.revenue,
                "sale_date": sale.sale_date,
            }
        )

    return pd.DataFrame(rows)


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
    payload = load_dataset_analysis()
    if payload:
        payload.setdefault("summary_cards", [])
        payload.setdefault("insights", {})
        if payload.get("retail_compatible"):
            retail_prediction = predict_next_month()
            if retail_prediction is not None and payload.get("summary_cards"):
                payload["summary_cards"][-1] = {
                    "label": "Next Month Revenue Forecast",
                    "value": format_display_value(retail_prediction),
                }
                current_note = payload["insights"].get("analysis_note", "")
                payload["insights"]["analysis_note"] = (
                    current_note + " Retail sales history is also available for next-month forecasting."
                ).strip()
        return payload

    sales_df = build_sales_dataframe()
    if sales_df is None:
        return build_empty_dashboard_payload()

    payload = analyze_uploaded_dataset(sales_df, "Retail Sales Records")
    retail_prediction = predict_next_month()

    if retail_prediction is not None and payload["summary_cards"]:
        payload["summary_cards"][-1] = {
            "label": "Next Month Revenue Forecast",
            "value": format_display_value(retail_prediction),
        }
        payload["insights"]["analysis_note"] += " Retail sales history is also available for next-month forecasting."

    return payload


def build_detected_fields(payload):
    insights = payload.get("insights", {})
    return [
        {"label": "Date Column", "value": insights.get("date_column", "Not detected")},
        {"label": "Metric Column", "value": insights.get("metric_column", "Not detected")},
        {"label": "Grouping Column", "value": insights.get("category_column", "Not detected")},
        {"label": "Top Segment", "value": insights.get("top_segment", "Not available")},
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


def import_retail_rows(df):
    working_df = df.copy()

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

    return render_template(
        "dashboard.html",
        user=current_user,
        active_dataset_name=payload.get("dataset_name", "Uploaded Dataset"),
        summary_cards=payload.get("summary_cards", []),
        trend_chart=payload.get("charts", {}).get("trend", {}),
        bar_chart=payload.get("charts", {}).get("breakdown", {}),
        pie_chart=payload.get("charts", {}).get("composition", {}),
        distribution_chart=payload.get("charts", {}).get("distribution", {}),
        analysis_note=payload.get("insights", {}).get("analysis_note", ""),
        detected_fields=build_detected_fields(payload),
        model_info=MODEL_INFO_CACHE,
        model_last_trained=MODEL_LAST_TRAINED,
    )


@main_routes.route("/upload", methods=["GET", "POST"])
@login_required
def upload_data():
    global MODEL_INFO_CACHE
    global MODEL_LAST_TRAINED

    if request.method == "POST":
        file = request.files["file"]

        if not file or not file.filename:
            flash("No file selected", "danger")
            return redirect(request.url)

        file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            df = load_uploaded_dataframe(file_path, file.filename)

            analysis = analyze_uploaded_dataset(df, file.filename)
            column_mapping = detect_columns(df.columns.tolist())
            analysis["retail_compatible"] = False
            save_dataset_analysis(analysis)

        except Exception as exc:
            flash(f"Error while reading the file: {str(exc)}", "danger")
            return redirect(request.url)

        if not column_mapping:
            flash("Dataset uploaded successfully. The dashboard is now analyzing it dynamically.", "success")
            flash("Retail forecasting was skipped because the file does not look like sales data.", "info")
            return redirect(url_for("main_routes.dashboard"))

        if len(df) > MAX_RETAIL_IMPORT_ROWS:
            flash("Large dataset uploaded successfully. The dashboard is analyzing it directly.", "success")
            flash(
                f"Retail database import was skipped because the file has {len(df):,} rows. "
                f"The app keeps dynamic analysis active for large files above {MAX_RETAIL_IMPORT_ROWS:,} rows.",
                "info",
            )
            return redirect(url_for("main_routes.dashboard"))

        retail_df = df.rename(columns=column_mapping).copy()

        try:
            imported_rows = import_retail_rows(retail_df)

            if imported_rows > 0:
                analysis["retail_compatible"] = True
                save_dataset_analysis(analysis)
                MODEL_INFO_CACHE = train_models()
                if MODEL_INFO_CACHE:
                    MODEL_LAST_TRAINED = datetime.now().strftime("%d %b %Y, %I:%M %p")

                flash("Dataset uploaded successfully. Dynamic analysis and retail forecasting are both ready.", "success")
            else:
                flash("Dataset uploaded successfully. Dynamic analysis is ready, but no valid sales rows were available for forecasting.", "warning")

        except Exception as exc:
            db.session.rollback()
            flash("Dataset uploaded successfully and generic analysis is ready.", "success")
            flash(f"Retail-specific import was skipped due to: {str(exc)}", "warning")

        return redirect(url_for("main_routes.dashboard"))

    return render_template("upload.html")


@main_routes.route("/train-model")
@login_required
def train_model_route():
    global MODEL_INFO_CACHE
    global MODEL_LAST_TRAINED

    MODEL_INFO_CACHE = train_models()

    if MODEL_INFO_CACHE:
        MODEL_LAST_TRAINED = datetime.now().strftime("%d %b %Y, %I:%M %p")
        flash("Model retrained successfully!", "success")
    else:
        flash("Not enough sales data to train the retail forecast model.", "danger")

    return redirect(url_for("main_routes.dashboard"))


@main_routes.route("/download-report")
@login_required
def download_report():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Retail Intelligence Dashboard Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="sales_prediction_report.pdf",
        mimetype="application/pdf",
    )
