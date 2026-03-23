from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file
from flask_login import login_required, current_user
from app.models import Store, Product, Sale
from app.analytics import get_dashboard_data
from app.ml_model import train_models, predict_next_month, category_wise_prediction
from app import db
import pandas as pd
import os
from datetime import datetime
from config import Config
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

main_routes = Blueprint("main_routes", __name__)

MODEL_INFO_CACHE = None
MODEL_LAST_TRAINED = None


def detect_columns(columns):
    """
    Auto-detect and map columns to required names.
    Returns a dict {original_col: required_col} or None if can't map.
    """
    mapping = {}
    columns_lower = [col.lower().strip() for col in columns]

    # Define possible names for each required column
    possible_names = {
        "store_name": ["store", "store_name", "shop", "branch", "store name", "shop name"],
        "city": ["city", "town", "location"],
        "state": ["state", "province", "region"],
        "product_name": ["product", "product_name", "item", "product name", "item name"],
        "category": ["category", "type", "class", "group"],
        "price": ["price", "unit_price", "cost", "unit price", "rate"],
        "quantity": ["quantity", "qty", "amount", "units", "count"],
        "revenue": ["revenue", "sales", "total", "amount", "value", "income"],
        "sale_date": ["date", "sale_date", "sale date", "transaction_date", "transaction date", "time", "datetime"]
    }

    for req, possibles in possible_names.items():
        for i, col_lower in enumerate(columns_lower):
            if any(poss in col_lower for poss in possibles):
                if columns[i] not in mapping.values():  # Avoid duplicate mapping
                    mapping[columns[i]] = req
                    break

    # Check if we have at least the essential ones
    essential = ["store_name", "product_name", "quantity", "revenue", "sale_date"]
    if not all(req in mapping.values() for req in essential):
        return None

    return mapping


@main_routes.route("/")
@login_required
def dashboard():
    global MODEL_INFO_CACHE
    global MODEL_LAST_TRAINED

    data = get_dashboard_data()
    prediction = predict_next_month()
    category_predictions, top_category = category_wise_prediction()

    sales = Sale.query.all()

    chart_labels = []
    chart_values = []
    category_labels = []
    category_values = []
    quantity_values = []

    if sales:
        df = pd.DataFrame([{
            "date": s.sale_date,
            "revenue": s.revenue,
            "quantity": s.quantity,
            "product_id": s.product_id
        } for s in sales])

        df["date"] = pd.to_datetime(df["date"])
        df["month_year"] = df["date"].dt.strftime("%b %Y")

        monthly = df.groupby("month_year")["revenue"].sum().reset_index()

        chart_labels = monthly["month_year"].tolist()
        chart_values = monthly["revenue"].tolist()

        if prediction:
            last_date = df["date"].max()
            next_month = (last_date + pd.DateOffset(months=1)).strftime("%b %Y")
            chart_labels.append(next_month)
            chart_values.append(prediction)

        # Merge with product categories
        product_data = pd.DataFrame([{
            "id": p.id,
            "category": p.category
        } for p in Product.query.all()])

        df = df.merge(product_data, left_on="product_id", right_on="id")

        category_group = df.groupby("category")["revenue"].sum().reset_index()

        category_labels = category_group["category"].tolist()
        category_values = category_group["revenue"].tolist()

        quantity_values = df["quantity"].tolist()

    return render_template(
        "dashboard.html",
        user=current_user,
        data=data,
        prediction=prediction,
        model_info=MODEL_INFO_CACHE,
        model_last_trained=MODEL_LAST_TRAINED,
        chart_labels=chart_labels,
        chart_values=chart_values,
        category_labels=category_labels,
        category_values=category_values,
        quantity_values=quantity_values,
        category_predictions=category_predictions,
        top_category=top_category
    )


@main_routes.route("/upload", methods=["GET", "POST"])
@login_required
def upload_data():
    if request.method == "POST":
        file = request.files["file"]

        if not file:
            flash("No file selected", "danger")
            return redirect(request.url)

        file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Auto-detect and map columns
            column_mapping = detect_columns(df.columns.tolist())
            if not column_mapping:
                flash("Could not detect required columns in the file. Please ensure your file contains sales data with columns for store, product, quantity, revenue, and date.", "danger")
                return redirect(request.url)

            df.rename(columns=column_mapping, inplace=True)

            # Check if all required are now present
            required_columns = ["store_name", "city", "state", "product_name", "category", "price", "quantity", "revenue", "sale_date"]
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                # For optional columns, fill with defaults
                if "city" in missing:
                    df["city"] = "Unknown"
                if "state" in missing:
                    df["state"] = "Unknown"
                if "category" in missing:
                    df["category"] = "General"
                if "price" in missing:
                    df["price"] = df["revenue"] / df["quantity"] if "quantity" in df.columns and df["quantity"].sum() > 0 else 0
                missing = [col for col in missing if col not in ["city", "state", "category", "price"]]
                if missing:
                    flash(f"Could not map required columns: {', '.join(missing)}", "danger")
                    return redirect(request.url)

            df.dropna(inplace=True)

            for _, row in df.iterrows():

                store = Store.query.filter_by(name=row["store_name"]).first()
                if not store:
                    store = Store(
                        name=row["store_name"],
                        city=row["city"],
                        state=row["state"]
                    )
                    db.session.add(store)
                    db.session.commit()

                product = Product.query.filter_by(name=row["product_name"]).first()
                if not product:
                    product = Product(
                        name=row["product_name"],
                        category=row["category"],
                        price=row["price"]
                    )
                    db.session.add(product)
                    db.session.commit()

                sale = Sale(
                    store_id=store.id,
                    product_id=product.id,
                    quantity=row["quantity"],
                    revenue=row["revenue"],
                    sale_date=pd.to_datetime(row["sale_date"])
                )

                db.session.add(sale)

            db.session.commit()
            flash("Data uploaded successfully!", "success")

            # Train model after upload
            MODEL_INFO_CACHE = train_models()
            if MODEL_INFO_CACHE:
                MODEL_LAST_TRAINED = datetime.now().strftime("%d %b %Y, %I:%M %p")

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

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
        flash("Not enough data to train model.", "danger")

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
        mimetype="application/pdf"
    )