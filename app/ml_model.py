import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from flask import current_app
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sqlalchemy import func

from app.analytics import (
    _coerce_numeric,
    _detect_date_column,
    _detect_metric_column,
    _latest_upload_file,
    _load_dataframe_from_file,
)
from app import db
from app.models import Sale
from app.utils import build_sales_dataframe

MODEL_FEATURE_VERSION = 2
TOP_CATEGORY_FEATURES = 2
TOP_STORE_FEATURES = 1
MIN_MONTHS_FOR_FORECAST = 4


def _model_path():
    return os.path.join(current_app.config["MODEL_FOLDER"], "best_model.pkl")


def _metadata_path():
    return os.path.join(current_app.config["MODEL_FOLDER"], "best_model_meta.json")


def _uploaded_forecast_cache_path():
    return os.path.join(current_app.config["MODEL_FOLDER"], "uploaded_forecast_meta.json")


def _uploaded_file_signature(file_path):
    if not file_path or not os.path.exists(file_path):
        return None

    stat = os.stat(file_path)
    return {
        "source_file": os.path.basename(file_path),
        "size": int(stat.st_size),
        "modified_at": int(stat.st_mtime),
    }


def _safe_r2_score(y_true, predictions):
    if len(y_true) < 2:
        return 0.0
    return round(float(r2_score(y_true, predictions)), 3)


def _feature_safe_name(value):
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in str(value).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def _wrap_legacy_artifact(artifact):
    model_name = getattr(getattr(artifact, "__class__", None), "__name__", "Legacy Model")
    return {
        "model": artifact,
        "best_model_name": model_name,
        "best_accuracy": None,
        "lr_accuracy": None,
        "rf_accuracy": None,
        "feature_columns": ["time_index"],
        "feature_count": 1,
        "feature_summary": "Legacy month index only",
        "feature_version": 1,
        "training_months": None,
        "sales_signature": None,
        "trained_at": None,
        "trained_at_label": "Legacy model",
        "is_legacy_artifact": True,
    }


def _sales_dataframe(dataframe=None):
    if dataframe is not None:
        working = dataframe.copy()
    else:
        working = build_sales_dataframe()

    if working.empty:
        return pd.DataFrame(columns=["sale_date", "revenue", "category", "store_name"])

    working["sale_date"] = pd.to_datetime(working["sale_date"], errors="coerce")
    working.dropna(subset=["sale_date"], inplace=True)
    working["revenue"] = pd.to_numeric(working["revenue"], errors="coerce").fillna(0.0)
    working.sort_values("sale_date", inplace=True)
    return working


def prepare_monthly_data(dataframe=None):
    working = _sales_dataframe(dataframe)
    if working.empty:
        return None

    working["month_end"] = working["sale_date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = (
        working.groupby("month_end", as_index=False)["revenue"]
        .sum()
        .sort_values("month_end")
    )

    if len(monthly) < 2:
        return None

    full_index = pd.date_range(
        start=monthly["month_end"].min(),
        end=monthly["month_end"].max(),
        freq="ME",
    )
    monthly = (
        monthly.set_index("month_end")
        .reindex(full_index, fill_value=0.0)
        .rename_axis("month_end")
        .reset_index()
    )
    monthly["time_index"] = range(1, len(monthly) + 1)
    return monthly


def _select_top_dimension_values(dataframe, column_name, limit):
    if dataframe.empty or column_name not in dataframe.columns:
        return []

    ranked = (
        dataframe.groupby(column_name)["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(limit)
    )
    return [value for value in ranked.index.tolist() if pd.notna(value)]


def _build_dimension_matrix(dataframe, column_name, selected_values, monthly_index, prefix):
    if dataframe.empty or not selected_values or column_name not in dataframe.columns:
        return pd.DataFrame(index=monthly_index)

    working = dataframe.copy()
    working["month_end"] = working["sale_date"].dt.to_period("M").dt.to_timestamp("M")
    pivot = working.pivot_table(
        index="month_end",
        columns=column_name,
        values="revenue",
        aggfunc="sum",
        fill_value=0.0,
    )

    matrix = pd.DataFrame(index=monthly_index)
    for value in selected_values:
        safe_name = f"{prefix}_{_feature_safe_name(value)}"
        if value in pivot.columns:
            matrix[safe_name] = pivot[value].reindex(monthly_index, fill_value=0.0).astype(float)
        else:
            matrix[safe_name] = 0.0

    return matrix


def _extend_revenue_series(monthly, periods_ahead=0):
    historical_index = pd.DatetimeIndex(monthly["month_end"])
    revenue_series = pd.Series(
        monthly["revenue"].astype(float).tolist(),
        index=historical_index,
        name="revenue",
    )

    if periods_ahead <= 0:
        return revenue_series, historical_index

    future_index = pd.date_range(
        start=historical_index[-1] + pd.offsets.MonthEnd(1),
        periods=periods_ahead,
        freq="ME",
    )
    full_index = historical_index.append(future_index)
    return revenue_series.reindex(full_index), historical_index


def _build_feature_frame(revenue_series, component_frames=None):
    frame = pd.DataFrame(index=revenue_series.index)
    frame["revenue"] = revenue_series.astype(float)
    frame["time_index"] = np.arange(1, len(frame) + 1, dtype=float)
    frame["year_offset"] = (frame.index.year - frame.index.year.min()).astype(float)
    frame["month"] = frame.index.month.astype(float)
    frame["quarter"] = frame.index.quarter.astype(float)
    frame["month_sin"] = np.sin((2 * np.pi * frame["month"]) / 12)
    frame["month_cos"] = np.cos((2 * np.pi * frame["month"]) / 12)

    previous_revenue = frame["revenue"].shift(1)
    frame["lag_1"] = previous_revenue
    frame["lag_2"] = frame["revenue"].shift(2)
    frame["lag_3"] = frame["revenue"].shift(3)
    frame["rolling_mean_3"] = previous_revenue.rolling(window=3, min_periods=1).mean()
    frame["rolling_mean_6"] = previous_revenue.rolling(window=6, min_periods=1).mean()
    frame["rolling_std_3"] = previous_revenue.rolling(window=3, min_periods=2).std()
    frame["momentum_1"] = frame["lag_1"] - frame["lag_2"]
    frame["growth_rate_1"] = (
        (frame["lag_1"] - frame["lag_2"]) / frame["lag_2"].replace(0, pd.NA)
    )

    if component_frames:
        previous_total = frame["lag_1"].replace(0, pd.NA)
        for component_frame in component_frames:
            if component_frame is None or component_frame.empty:
                continue

            for column_name in component_frame.columns:
                lagged_component = component_frame[column_name].shift(1)
                frame[f"{column_name}_lag1"] = lagged_component
                frame[f"{column_name}_share_lag1"] = lagged_component / previous_total

    feature_columns = [column for column in frame.columns if column != "revenue"]
    frame[feature_columns] = (
        frame[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return frame


def _build_multivariate_training_frames(dataframe=None, periods_ahead=1):
    working = _sales_dataframe(dataframe)
    monthly = prepare_monthly_data(working)
    if monthly is None or len(monthly) < MIN_MONTHS_FOR_FORECAST:
        return None

    revenue_series, historical_index = _extend_revenue_series(monthly, periods_ahead=periods_ahead)
    full_index = revenue_series.index

    top_categories = _select_top_dimension_values(working, "category", TOP_CATEGORY_FEATURES)
    top_stores = _select_top_dimension_values(working, "store_name", TOP_STORE_FEATURES)

    category_matrix = _build_dimension_matrix(
        working,
        "category",
        top_categories,
        full_index,
        "category",
    )
    store_matrix = _build_dimension_matrix(
        working,
        "store_name",
        top_stores,
        full_index,
        "store",
    )

    feature_frame = _build_feature_frame(revenue_series, [category_matrix, store_matrix])
    historical_count = len(historical_index)

    return {
        "historical": feature_frame.iloc[:historical_count].copy(),
        "future": feature_frame.iloc[historical_count:].copy(),
        "top_categories": [str(value) for value in top_categories],
        "top_stores": [str(value) for value in top_stores],
    }


def _build_univariate_training_frames(monthly, periods_ahead=1):
    if monthly is None or len(monthly) < MIN_MONTHS_FOR_FORECAST:
        return None

    revenue_series, historical_index = _extend_revenue_series(monthly, periods_ahead=periods_ahead)
    feature_frame = _build_feature_frame(revenue_series)
    historical_count = len(historical_index)

    return {
        "historical": feature_frame.iloc[:historical_count].copy(),
        "future": feature_frame.iloc[historical_count:].copy(),
    }


def get_sales_signature():
    result = db.session.query(
        func.count(Sale.id),
        func.max(Sale.id),
        func.max(Sale.sale_date),
        func.coalesce(func.sum(Sale.revenue), 0.0),
    ).first()

    count, latest_id, latest_date, total_revenue = result

    if not count:
        return {
            "count": 0,
            "latest_sale_id": 0,
            "latest_sale_date": None,
            "total_revenue": 0.0,
        }

    return {
        "count": count,
        "latest_sale_id": latest_id,
        "latest_sale_date": latest_date.isoformat() if latest_date else None,
        "total_revenue": round(float(total_revenue), 2),
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
    if len(features) >= 6:
        evaluation_size = min(3, max(2, len(features) // 4))
        split_index = len(features) - evaluation_size
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


def _candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=1,
            random_state=42,
        ),
    }


def _train_best_model(training_frame):
    feature_columns = [column for column in training_frame.columns if column != "revenue"]
    features = training_frame[feature_columns]
    target = training_frame["revenue"]

    candidate_models = _candidate_models()
    scores = {}
    for model_name, model in candidate_models.items():
        scores[model_name] = _evaluate_candidate(model, features, target)

    best_model_name = max(scores, key=scores.get)
    best_model = candidate_models[best_model_name]
    best_model.fit(features, target)

    return {
        "model": best_model,
        "feature_columns": feature_columns,
        "scores": scores,
        "best_model_name": best_model_name,
    }


def train_models(force=False):
    os.makedirs(current_app.config["MODEL_FOLDER"], exist_ok=True)

    training_payload = _build_multivariate_training_frames(periods_ahead=1)
    if training_payload is None:
        return None

    current_signature = get_sales_signature()
    existing_metadata = get_model_status()
    if (
        not force
        and existing_metadata
        and existing_metadata.get("sales_signature") == current_signature
        and existing_metadata.get("feature_version") == MODEL_FEATURE_VERSION
        and os.path.exists(_model_path())
    ):
        return existing_metadata

    training_result = _train_best_model(training_payload["historical"])
    trained_at = datetime.now()

    top_categories = training_payload["top_categories"]
    top_stores = training_payload["top_stores"]
    feature_summary_parts = [
        "calendar seasonality",
        "lagged monthly revenue",
        "rolling averages",
        "revenue momentum",
    ]
    if top_categories:
        feature_summary_parts.append(f"top categories ({', '.join(top_categories)})")
    if top_stores:
        feature_summary_parts.append(f"top stores ({', '.join(top_stores)})")

    artifact = {
        "model": training_result["model"],
        "best_model_name": training_result["best_model_name"],
        "best_accuracy": round(float(training_result["scores"][training_result["best_model_name"]]), 3),
        "lr_accuracy": round(float(training_result["scores"]["Linear Regression"]), 3),
        "rf_accuracy": round(float(training_result["scores"]["Random Forest"]), 3),
        "feature_columns": training_result["feature_columns"],
        "feature_count": len(training_result["feature_columns"]),
        "feature_summary": ", ".join(feature_summary_parts),
        "feature_version": MODEL_FEATURE_VERSION,
        "training_months": int(len(training_payload["historical"])),
        "top_categories": top_categories,
        "top_stores": top_stores,
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
    current_signature = get_sales_signature()

    should_retrain = (
        force
        or metadata is None
        or metadata.get("sales_signature") != current_signature
        or metadata.get("feature_version") != MODEL_FEATURE_VERSION
        or not os.path.exists(_model_path())
    )

    if should_retrain:
        return train_models(force=True)
    return metadata


def predict_next_month():
    metadata = sync_model_with_sales_data()
    if metadata is None:
        return None

    artifact = load_model_artifact()
    if artifact is None:
        return None

    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns") or []
    if not hasattr(model, "predict") or not feature_columns:
        return None

    training_payload = _build_multivariate_training_frames(periods_ahead=1)
    if training_payload is None or training_payload["future"].empty:
        return None

    future_features = training_payload["future"]
    missing_columns = [column for column in feature_columns if column not in future_features.columns]
    if missing_columns:
        metadata = sync_model_with_sales_data(force=True)
        artifact = load_model_artifact()
        if artifact is None:
            return None

        model = artifact.get("model")
        feature_columns = artifact.get("feature_columns") or []
        future_features = _build_multivariate_training_frames(periods_ahead=1)
        if future_features is None or future_features["future"].empty:
            return None
        future_features = future_features["future"]

    prediction_frame = future_features[feature_columns].iloc[[0]]
    prediction = model.predict(prediction_frame)
    return round(float(prediction[0]), 2)


def _uploaded_metric_dataframe():
    latest_file = _latest_upload_file()
    if not latest_file:
        return None

    dataframe = _load_dataframe_from_file(latest_file)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    date_column = _detect_date_column(dataframe)
    metric_column = _detect_metric_column(dataframe, excluded_columns={date_column} if date_column else set())
    if not date_column or not metric_column:
        return None

    working = pd.DataFrame(
        {
            "sale_date": pd.to_datetime(dataframe[date_column], errors="coerce"),
            "revenue": _coerce_numeric(dataframe[metric_column]),
            "category": "Uploaded Dataset",
            "store_name": "Uploaded Dataset",
        }
    )
    working.dropna(subset=["sale_date", "revenue"], inplace=True)
    if working.empty:
        return None

    return {
        "dataframe": working,
        "date_column": date_column,
        "metric_column": metric_column,
        "source_file": os.path.basename(latest_file),
    }


def get_cached_uploaded_prediction():
    latest_file = _latest_upload_file()
    signature = _uploaded_file_signature(latest_file)
    cache_path = _uploaded_forecast_cache_path()
    if not signature or not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
        if cached.get("file_signature") == signature:
            return cached.get("forecast")
    except Exception:
        return None

    return None


def predict_next_month_from_uploaded_dataset():
    cached_prediction = get_cached_uploaded_prediction()
    if cached_prediction is not None:
        return cached_prediction

    latest_file = _latest_upload_file()
    signature = _uploaded_file_signature(latest_file)
    source = _uploaded_metric_dataframe()
    if source is None:
        return None

    monthly = prepare_monthly_data(source["dataframe"])
    prediction = _forecast_group_series(monthly)
    if prediction is None:
        return None

    forecast = {
        "prediction": prediction,
        "metric_column": source["metric_column"],
        "date_column": source["date_column"],
        "source_file": source["source_file"],
        "training_months": int(len(monthly)),
    }

    if signature:
        os.makedirs(current_app.config["MODEL_FOLDER"], exist_ok=True)
        with open(_uploaded_forecast_cache_path(), "w", encoding="utf-8") as cache_file:
            json.dump({"file_signature": signature, "forecast": forecast}, cache_file, indent=2)

    return forecast


def _forecast_group_series(monthly):
    training_payload = _build_univariate_training_frames(monthly, periods_ahead=1)
    if training_payload is None or training_payload["future"].empty:
        return None

    training_result = _train_best_model(training_payload["historical"])
    prediction_frame = training_payload["future"][training_result["feature_columns"]].iloc[[0]]
    prediction = training_result["model"].predict(prediction_frame)
    return round(float(prediction[0]), 2)


def category_wise_prediction():
    dataframe = _sales_dataframe()
    if dataframe.empty or "category" not in dataframe.columns:
        return None, None

    category_predictions = {}
    for category, category_frame in dataframe.groupby("category"):
        monthly = prepare_monthly_data(category_frame)
        prediction = _forecast_group_series(monthly)
        if prediction is None:
            continue
        category_predictions[str(category)] = prediction

    if not category_predictions:
        return None, None

    top_category = max(category_predictions, key=category_predictions.get)
    return category_predictions, top_category
