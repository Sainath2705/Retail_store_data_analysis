import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from app.models import Sale, Product
from config import Config

MODEL_PATH = os.path.join(Config.MODEL_FOLDER, "best_model.pkl")


def prepare_monthly_data():
    sales = Sale.query.all()

    if len(sales) < 3:
        return None

    df = pd.DataFrame([{
        "date": s.sale_date,
        "revenue": s.revenue
    } for s in sales])

    df["date"] = pd.to_datetime(df["date"])
    df["month_year"] = df["date"].dt.to_period("M")

    monthly = df.groupby("month_year")["revenue"].sum().reset_index()

    if len(monthly) < 2:
        return None

    monthly["time_index"] = range(1, len(monthly) + 1)

    return monthly


def train_models():
    monthly = prepare_monthly_data()

    if monthly is None:
        return None

    X = monthly[["time_index"]]
    y = monthly["revenue"]

    lr_model = LinearRegression()
    lr_model.fit(X, y)
    lr_accuracy = r2_score(y, lr_model.predict(X))

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_accuracy = r2_score(y, rf_model.predict(X))

    if rf_accuracy > lr_accuracy:
        best_model = rf_model
        best_name = "Random Forest"
        best_accuracy = rf_accuracy
    else:
        best_model = lr_model
        best_name = "Linear Regression"
        best_accuracy = lr_accuracy

    joblib.dump(best_model, MODEL_PATH)

    return {
        "lr_accuracy": round(float(lr_accuracy), 3),
        "rf_accuracy": round(float(rf_accuracy), 3),
        "best_model_name": best_name,
        "best_accuracy": round(float(best_accuracy), 3)
    }


def predict_next_month():
    if not os.path.exists(MODEL_PATH):
        return None

    monthly = prepare_monthly_data()
    if monthly is None:
        return None

    model = joblib.load(MODEL_PATH)
    next_index = len(monthly) + 1
    prediction = model.predict([[next_index]])

    return round(float(prediction[0]), 2)


# ---------------- CATEGORY PREDICTION ----------------

def category_wise_prediction():
    sales = Sale.query.all()

    if not sales:
        return None, None

    data = []

    for s in sales:
        product = Product.query.get(s.product_id)
        data.append({
            "date": s.sale_date,
            "revenue": s.revenue,
            "category": product.category
        })

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["month_year"] = df["date"].dt.to_period("M")

    category_predictions = {}

    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        monthly = cat_df.groupby("month_year")["revenue"].sum().reset_index()

        if len(monthly) < 2:
            continue

        monthly["time_index"] = range(1, len(monthly) + 1)

        X = monthly[["time_index"]]
        y = monthly["revenue"]

        model = LinearRegression()
        model.fit(X, y)

        next_index = len(monthly) + 1
        prediction = model.predict([[next_index]])

        category_predictions[category] = round(float(prediction[0]), 2)

    if not category_predictions:
        return None, None

    top_category = max(category_predictions, key=category_predictions.get)

    return category_predictions, top_category