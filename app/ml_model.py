import json
import os
import pickle
from datetime import datetime

import pandas as pd
from flask import current_app
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from app.models import Product, Sale


def _model_path():
    return os.path.join(current_app.config["MODEL_FOLDER"], "best_model.pkl")


def _metadata_path():
    return os.path.join(current_app.config["MODEL_FOLDER"], "best_model_meta.json")


def _safe_r2_score(y_true, predictions):
    if len(y_true) < 2:
        return 0.0
    return round(float(r2_score(y_true, predictions)), 3)


def _wrap_legacy_artifact(artifact):
    model_name = getattr(getattr(artifact, "__class__", None), "__name__", "Legacy Model")
    return {
        "model": artifact,
        "best_model_name": model_name,
        "best_accuracy": None,
        "lr_accuracy": None,
        "rf_accuracy": None,
        "sales_signature": None,
        "trained_at": None,
        "trained_at_label": "Legacy model",
        "is_legacy_artifact": True,
    }


def prepare_monthly_data():
    sales = Sale.query.order_by(Sale.sale_date.asc()).all()
    if len(sales) < 3:
        return None

    dataframe = pd.DataFrame(
        [{"date": sale.sale_date, "revenue": sale.revenue} for sale in sales]
    )
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    dataframe.dropna(subset=["date"], inplace=True)
    dataframe["month_year"] = dataframe["date"].dt.to_period("M")

    monthly = dataframe.groupby("month_year")["revenue"].sum().reset_index()
    if len(monthly) < 2:
        return None

    monthly["time_index"] = range(1, len(monthly) + 1)
    return monthly


def get_sales_signature():
    sales = Sale.query.order_by(Sale.id.asc()).all()
    if not sales:
        return {
            "count": 0,
            "latest_sale_id": 0,
            "latest_sale_date": None,
            "total_revenue": 0.0,
        }

    latest_sale = sales[-1]
    return {
        "count": len(sales),
        "latest_sale_id": latest_sale.id,
        "latest_sale_date": latest_sale.sale_date.isoformat() if latest_sale.sale_date else None,
        "total_revenue": round(sum(float(sale.revenue or 0) for sale in sales), 2),
    }


def load_model_artifact():
    model_path = _model_path()
    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as model_file:
        artifact = pickle.load(model_file)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact

    return _wrap_legacy_artifact(artifact)


def get_model_status():
    metadata_path = _metadata_path()
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    artifact = load_model_artifact()
    if artifact is None:
        return None

    return {key: value for key, value in artifact.items() if key != "model"}


def _evaluate_candidate(model, features, target):
    if len(features) >= 4:
        split_index = max(2, int(len(features) * 0.8))
        if split_index >= len(features):
            split_index = len(features) - 1

        x_train = features.iloc[:split_index]
        y_train = target.iloc[:split_index]
        x_eval = features.iloc[split_index:]
        y_eval = target.iloc[split_index:]
    else:
        x_train = features
        y_train = target
        x_eval = features
        y_eval = target

    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)
    return _safe_r2_score(y_eval, predictions)


def train_models(force=False):
    os.makedirs(current_app.config["MODEL_FOLDER"], exist_ok=True)

    monthly = prepare_monthly_data()
    if monthly is None:
        return None

    current_signature = get_sales_signature()
    existing_metadata = get_model_status()
    if (
        not force
        and existing_metadata
        and existing_metadata.get("sales_signature") == current_signature
        and os.path.exists(_model_path())
    ):
        return existing_metadata

    features = monthly[["time_index"]]
    target = monthly["revenue"]

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    scores = {}
    for model_name, model in candidate_models.items():
        scores[model_name] = _evaluate_candidate(model, features, target)

    best_model_name = max(scores, key=scores.get)
    best_model = candidate_models[best_model_name]
    best_model.fit(features, target)

    trained_at = datetime.now()
    artifact = {
        "model": best_model,
        "best_model_name": best_model_name,
        "best_accuracy": round(float(scores[best_model_name]), 3),
        "lr_accuracy": round(float(scores["Linear Regression"]), 3),
        "rf_accuracy": round(float(scores["Random Forest"]), 3),
        "sales_signature": current_signature,
        "trained_at": trained_at.isoformat(),
        "trained_at_label": trained_at.strftime("%d %b %Y, %I:%M %p"),
    }

    with open(_model_path(), "wb") as model_file:
        pickle.dump(artifact, model_file)

    metadata = {key: value for key, value in artifact.items() if key != "model"}
    with open(_metadata_path(), "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return metadata


def sync_model_with_sales_data(force=False):
    metadata = get_model_status()
    if force or metadata is None or metadata.get("sales_signature") != get_sales_signature():
        return train_models(force=True)
    return metadata


def predict_next_month():
    artifact = load_model_artifact()
    if artifact is None:
        return None

    monthly = prepare_monthly_data()
    if monthly is None:
        return None

    model = artifact.get("model")
    if not hasattr(model, "predict"):
        return None

    next_index = len(monthly) + 1
    prediction = model.predict([[next_index]])
    return round(float(prediction[0]), 2)


def category_wise_prediction():
    sales = Sale.query.all()
    if not sales:
        return None, None

    product_map = {product.id: product for product in Product.query.all()}
    data = []
    for sale in sales:
        product = product_map.get(sale.product_id)
        data.append(
            {
                "date": sale.sale_date,
                "revenue": sale.revenue,
                "category": product.category if product else "General",
            }
        )

    dataframe = pd.DataFrame(data)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    dataframe.dropna(subset=["date"], inplace=True)
    dataframe["month_year"] = dataframe["date"].dt.to_period("M")

    category_predictions = {}
    for category in dataframe["category"].unique():
        category_frame = dataframe[dataframe["category"] == category]
        monthly = category_frame.groupby("month_year")["revenue"].sum().reset_index()
        if len(monthly) < 2:
            continue

        monthly["time_index"] = range(1, len(monthly) + 1)
        features = monthly[["time_index"]]
        target = monthly["revenue"]

        model = LinearRegression()
        model.fit(features, target)

        next_index = len(monthly) + 1
        prediction = model.predict([[next_index]])
        category_predictions[category] = round(float(prediction[0]), 2)

    if not category_predictions:
        return None, None

    top_category = max(category_predictions, key=category_predictions.get)
    return category_predictions, top_category
