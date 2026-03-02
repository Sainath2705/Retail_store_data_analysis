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

    if sales:
        df = pd.DataFrame([{
            "date": s.sale_date,
            "revenue": s.revenue
        } for s in sales])

        df["date"] = pd.to_datetime(df["date"])
        df["month_year"] = df["date"].dt.strftime("%b %Y")

        monthly = df.groupby("month_year")["revenue"].sum().reset_index()

        chart_labels = monthly["month_year"].tolist()
        chart_values = monthly["revenue"].tolist()

        if prediction:
            last_date = pd.to_datetime(df["date"]).max()
            next_month = (last_date + pd.DateOffset(months=1)).strftime("%b %Y")

            chart_labels.append(next_month)
            chart_values.append(prediction)

    return render_template(
        "dashboard.html",
        user=current_user,
        data=data,
        prediction=prediction,
        model_info=MODEL_INFO_CACHE,
        model_last_trained=MODEL_LAST_TRAINED,
        chart_labels=chart_labels,
        chart_values=chart_values,
        category_predictions=category_predictions,
        top_category=top_category
    )


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
    data = get_dashboard_data()
    prediction = predict_next_month()
    category_predictions, top_category = category_wise_prediction()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Retail Intelligence Dashboard Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))

    if data:
        elements.append(Paragraph(f"Total Revenue: ₹{data['total_revenue']}", styles["Normal"]))
        elements.append(Paragraph(f"Total Orders: {data['total_orders']}", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

    if prediction:
        elements.append(Paragraph(f"Next Month Prediction: ₹{prediction}", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

    if MODEL_LAST_TRAINED:
        elements.append(Paragraph(f"Model Last Trained: {MODEL_LAST_TRAINED}", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

    if category_predictions:
        elements.append(Paragraph("<b>Category Forecast:</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        for category, value in category_predictions.items():
            elements.append(Paragraph(f"{category}: ₹{value}", styles["Normal"]))

        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"Top Performing Category: {top_category}", styles["Normal"]))

    doc.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="sales_prediction_report.pdf",
        mimetype="application/pdf"
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

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

        return redirect(url_for("main_routes.dashboard"))

    return render_template("upload.html")