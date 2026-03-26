import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.models import Product, Sale
from config import Config

DATASET_ANALYSIS_PATH = os.path.join(
    Config.UPLOAD_FOLDER,
    "latest_dataset_analysis.json",
)

NUMERIC_PRIORITY_TOKENS = (
    "revenue",
    "sales",
    "amount",
    "total",
    "value",
    "price",
    "profit",
    "cost",
    "score",
    "quantity",
    "qty",
    "count",
)

CATEGORY_PRIORITY_TOKENS = (
    "category",
    "type",
    "class",
    "group",
    "segment",
    "status",
    "region",
    "city",
    "state",
    "product",
    "item",
    "store",
    "branch",
    "name",
)

DATE_TOKENS = ("date", "time", "day", "month", "year")


def get_dashboard_data():
    sales = Sale.query.all()

    if not sales:
        return None

    data = []

    for sale in sales:
        product = Product.query.get(sale.product_id)
        data.append(
            {
                "product": product.name,
                "category": product.category,
                "quantity": sale.quantity,
                "revenue": sale.revenue,
                "date": sale.sale_date,
            }
        )

    df = pd.DataFrame(data)

    total_revenue = float(df["revenue"].sum())
    total_orders = int(len(df))

    category_revenue = df.groupby("category")["revenue"].sum().to_dict()
    product_revenue = df.groupby("product")["revenue"].sum().to_dict()

    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly_trend = df.groupby("month")["revenue"].sum().astype(float).to_dict()

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "category_revenue": category_revenue,
        "product_revenue": product_revenue,
        "monthly_trend": {str(key): value for key, value in monthly_trend.items()},
    }


def analyze_uploaded_dataset(df, source_name):
    working_df = df.copy()
    working_df.columns = [str(column).strip() for column in working_df.columns]

    row_count = int(len(working_df))
    column_count = int(len(working_df.columns))
    missing_cells = int(working_df.isna().sum().sum())

    working_df, date_columns = infer_date_columns(working_df)
    numeric_columns = get_numeric_columns(working_df)

    metric_column = select_metric_column(working_df, numeric_columns)
    date_column = select_date_column(date_columns)
    category_column = select_category_column(
        working_df,
        excluded={metric_column, date_column},
    )

    trend_chart = build_trend_chart(working_df, date_column, metric_column)
    breakdown_chart = build_breakdown_chart(
        working_df,
        category_column,
        metric_column,
    )
    pie_chart = {
        "title": breakdown_chart["title"],
        "labels": breakdown_chart["labels"],
        "values": breakdown_chart["values"],
        "dataset_label": breakdown_chart["dataset_label"],
    }
    distribution_chart = build_distribution_chart(working_df, metric_column)
    forecast = build_forecast(trend_chart)

    primary_label, primary_value = build_primary_summary(working_df, metric_column)
    summary_cards = [
        {
            "label": primary_label,
            "value": format_display_value(primary_value),
        },
        {
            "label": "Rows",
            "value": format_display_value(row_count),
        },
        {
            "label": "Columns",
            "value": format_display_value(column_count),
        },
    ]

    if forecast:
        summary_cards.append(
            {
                "label": forecast["card_label"],
                "value": forecast["display_value"],
            }
        )
    else:
        summary_cards.append(
            {
                "label": "Missing Cells",
                "value": format_display_value(missing_cells),
            }
        )

    insights = build_insights(
        date_column=date_column,
        metric_column=metric_column,
        category_column=category_column,
        breakdown_chart=breakdown_chart,
        forecast=forecast,
    )

    return {
        "dataset_name": source_name,
        "row_count": row_count,
        "column_count": column_count,
        "summary_cards": summary_cards,
        "charts": {
            "trend": trend_chart,
            "breakdown": breakdown_chart,
            "composition": pie_chart,
            "distribution": distribution_chart,
        },
        "insights": insights,
    }


def save_dataset_analysis(analysis):
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    with open(DATASET_ANALYSIS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(analysis, file_obj, indent=2)


def load_dataset_analysis():
    if not os.path.exists(DATASET_ANALYSIS_PATH):
        return None

    with open(DATASET_ANALYSIS_PATH, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def infer_date_columns(df):
    working_df = df.copy()
    detected_columns = []

    for column in working_df.columns:
        series = working_df[column]
        column_name = normalize_name(column)

        should_try_parse = (
            pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or any(token in column_name for token in DATE_TOKENS)
        )

        if not should_try_parse:
            continue

        parsed = pd.to_datetime(series, errors="coerce")
        valid_ratio = parsed.notna().mean() if len(parsed) else 0

        if valid_ratio >= 0.6 and parsed.nunique(dropna=True) >= 2:
            working_df[column] = parsed
            detected_columns.append(column)

    return working_df, detected_columns


def get_numeric_columns(df):
    numeric_columns = []

    for column in df.columns:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        if series.dropna().empty:
            continue

        numeric_columns.append(column)

    return numeric_columns


def select_metric_column(df, numeric_columns):
    candidates = []

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        valid_values = series.dropna()

        if len(valid_values) < 2:
            continue

        score = 0
        column_name = normalize_name(column)

        if looks_like_identifier(valid_values, column_name):
            score -= 30

        if valid_values.nunique() <= 2:
            score -= 10

        score += valid_values.nunique() * 0.2
        score += valid_values.notna().mean() * 10

        if any(token in column_name for token in NUMERIC_PRIORITY_TOKENS):
            score += 20

        candidates.append((score, column))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def select_date_column(date_columns):
    if not date_columns:
        return None

    ranked_columns = sorted(
        date_columns,
        key=lambda value: (
            not any(token in normalize_name(value) for token in DATE_TOKENS),
            value,
        ),
    )
    return ranked_columns[0]


def select_category_column(df, excluded):
    candidates = []

    for column in df.columns:
        if column in excluded:
            continue

        series = df[column]
        column_name = normalize_name(column)

        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        valid_values = series.dropna().astype(str).str.strip()
        valid_values = valid_values[valid_values != ""]

        if valid_values.empty:
            continue

        unique_count = valid_values.nunique()
        unique_ratio = unique_count / max(len(valid_values), 1)
        score = 0

        if looks_like_identifier(valid_values, column_name):
            score -= 25

        if any(token in column_name for token in CATEGORY_PRIORITY_TOKENS):
            score += 20

        if 2 <= unique_count <= 20:
            score += 15
        elif unique_count <= 50:
            score += 5
        else:
            score -= 10

        if unique_ratio > 0.8:
            score -= 10

        candidates.append((score, column))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def build_trend_chart(df, date_column, metric_column):
    if date_column and metric_column:
        series = build_time_series(df, date_column, metric_column)
        if series["labels"]:
            return {
                "title": f"{make_label(metric_column)} Trend",
                "labels": series["labels"],
                "values": series["values"],
                "dataset_label": f"Total {make_label(metric_column)}",
                "next_label": series["next_label"],
                "granularity": series["granularity"],
            }

    if metric_column:
        metric_values = pd.to_numeric(df[metric_column], errors="coerce").dropna().head(20)
        labels = [f"Row {index}" for index in range(1, len(metric_values) + 1)]
        return {
            "title": f"{make_label(metric_column)} Sample",
            "labels": labels,
            "values": [round(float(value), 2) for value in metric_values.tolist()],
            "dataset_label": make_label(metric_column),
            "next_label": None,
            "granularity": None,
        }

    return empty_chart("Dataset Trend")


def build_time_series(df, date_column, metric_column):
    working_df = df[[date_column, metric_column]].copy()
    working_df[metric_column] = pd.to_numeric(working_df[metric_column], errors="coerce")
    working_df.dropna(subset=[date_column, metric_column], inplace=True)

    if len(working_df) < 2:
        return {"labels": [], "values": [], "next_label": None, "granularity": None}

    working_df.sort_values(date_column, inplace=True)

    span_days = int((working_df[date_column].max() - working_df[date_column].min()).days)
    unique_days = working_df[date_column].dt.normalize().nunique()

    if span_days > 120 or unique_days > 31:
        grouped = working_df.groupby(working_df[date_column].dt.to_period("M"))[metric_column].sum()
        labels = [period.strftime("%b %Y") for period in grouped.index]
        next_label = (grouped.index[-1] + 1).strftime("%b %Y")
        granularity = "month"
    elif span_days > 21 or unique_days > 10:
        grouped = working_df.groupby(working_df[date_column].dt.to_period("W"))[metric_column].sum()
        labels = [f"Week of {period.start_time.strftime('%d %b %Y')}" for period in grouped.index]
        next_label = f"Week of {(grouped.index[-1] + 1).start_time.strftime('%d %b %Y')}"
        granularity = "week"
    else:
        grouped = working_df.groupby(working_df[date_column].dt.normalize())[metric_column].sum()
        labels = [timestamp.strftime("%d %b %Y") for timestamp in grouped.index]
        next_label = (grouped.index[-1] + pd.Timedelta(days=1)).strftime("%d %b %Y")
        granularity = "day"

    values = [round(float(value), 2) for value in grouped.tolist()]
    return {
        "labels": labels,
        "values": values,
        "next_label": next_label,
        "granularity": granularity,
    }


def build_breakdown_chart(df, category_column, metric_column):
    if category_column and metric_column:
        working_df = df[[category_column, metric_column]].copy()
        working_df[metric_column] = pd.to_numeric(working_df[metric_column], errors="coerce")
        working_df.dropna(subset=[category_column, metric_column], inplace=True)

        if not working_df.empty:
            grouped = (
                working_df.groupby(category_column)[metric_column]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            return {
                "title": f"{make_label(metric_column)} by {make_label(category_column)}",
                "labels": [str(value) for value in grouped.index.tolist()],
                "values": [round(float(value), 2) for value in grouped.tolist()],
                "dataset_label": f"Total {make_label(metric_column)}",
            }

    if category_column:
        valid_values = (
            df[category_column]
            .dropna()
            .astype(str)
            .str.strip()
        )
        valid_values = valid_values[valid_values != ""]

        if not valid_values.empty:
            grouped = valid_values.value_counts().head(10)
            return {
                "title": f"Top {make_label(category_column)} Values",
                "labels": grouped.index.tolist(),
                "values": grouped.astype(int).tolist(),
                "dataset_label": "Record Count",
            }

    completeness = df.notna().sum().sort_values(ascending=False).head(10)
    return {
        "title": "Column Completeness",
        "labels": completeness.index.tolist(),
        "values": completeness.astype(int).tolist(),
        "dataset_label": "Non-Null Values",
    }


def build_distribution_chart(df, metric_column):
    if not metric_column:
        return empty_chart("Distribution")

    values = pd.to_numeric(df[metric_column], errors="coerce").dropna()

    if values.empty:
        return empty_chart(f"{make_label(metric_column)} Distribution")

    if values.nunique() == 1:
        return {
            "title": f"{make_label(metric_column)} Distribution",
            "labels": [format_display_value(values.iloc[0])],
            "values": [int(len(values))],
            "dataset_label": "Record Count",
        }

    bin_count = min(10, max(4, int(np.sqrt(len(values)))))
    buckets = pd.cut(values, bins=bin_count, include_lowest=True, duplicates="drop")
    grouped = buckets.value_counts(sort=False)

    labels = [
        f"{format_short_number(interval.left)} to {format_short_number(interval.right)}"
        for interval in grouped.index
    ]

    return {
        "title": f"{make_label(metric_column)} Distribution",
        "labels": labels,
        "values": grouped.astype(int).tolist(),
        "dataset_label": "Record Count",
    }


def build_forecast(trend_chart):
    values = trend_chart.get("values", [])
    next_label = trend_chart.get("next_label")
    granularity = trend_chart.get("granularity")

    if len(values) < 3 or not next_label or not granularity:
        return None

    x_values = np.arange(1, len(values) + 1).reshape(-1, 1)
    y_values = np.array(values, dtype=float)

    model = LinearRegression()
    model.fit(x_values, y_values)

    predicted_value = round(float(model.predict([[len(values) + 1]])[0]), 2)

    return {
        "card_label": f"Next {granularity.title()} Forecast",
        "display_value": format_display_value(predicted_value),
        "value": predicted_value,
        "next_label": next_label,
    }


def build_primary_summary(df, metric_column):
    if not metric_column:
        return "Records", len(df)

    values = pd.to_numeric(df[metric_column], errors="coerce").dropna()
    if values.empty:
        return "Records", len(df)

    return f"Total {make_label(metric_column)}", round(float(values.sum()), 2)


def build_insights(date_column, metric_column, category_column, breakdown_chart, forecast):
    top_segment = None
    if breakdown_chart["labels"]:
        top_segment = breakdown_chart["labels"][0]

    note_parts = []

    if date_column and metric_column:
        note_parts.append(
            f"Using {make_label(date_column)} as the timeline and {make_label(metric_column)} as the main metric."
        )
    elif metric_column:
        note_parts.append(
            f"Using {make_label(metric_column)} as the main numeric field because no reliable timeline was detected."
        )
    else:
        note_parts.append(
            "No strong numeric measure was detected, so the dashboard falls back to structural summaries."
        )

    if category_column:
        note_parts.append(f"Grouping records by {make_label(category_column)} where possible.")

    if forecast:
        note_parts.append(f"A simple forward forecast is shown for {forecast['next_label']}.")

    return {
        "analysis_note": " ".join(note_parts),
        "date_column": make_label(date_column) if date_column else "Not detected",
        "metric_column": make_label(metric_column) if metric_column else "Not detected",
        "category_column": make_label(category_column) if category_column else "Not detected",
        "top_segment": top_segment or "Not available",
    }


def looks_like_identifier(series, column_name):
    unique_count = series.nunique(dropna=True)
    total_count = len(series.dropna())

    if total_count == 0:
        return False

    if "id" in column_name or column_name.endswith("code"):
        return unique_count >= total_count * 0.9

    return unique_count >= total_count * 0.98 and unique_count > 20


def empty_chart(title):
    return {
        "title": title,
        "labels": [],
        "values": [],
        "dataset_label": "",
        "next_label": None,
        "granularity": None,
    }


def make_label(value):
    if not value:
        return "Value"
    return str(value).replace("_", " ").strip().title()


def normalize_name(value):
    return str(value).strip().lower().replace("-", " ").replace("_", " ")


def format_display_value(value):
    if value is None:
        return "Not available"

    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"

    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return f"{int(numeric_value):,}"
        return f"{numeric_value:,.2f}"

    return str(value)


def format_short_number(value):
    numeric_value = float(value)
    if numeric_value.is_integer():
        return f"{int(numeric_value):,}"
    return f"{numeric_value:,.2f}"
