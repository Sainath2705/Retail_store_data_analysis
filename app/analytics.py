import json
import os
import re
from typing import Optional

import pandas as pd
from flask import current_app

ANALYSIS_CACHE_VERSION = 2

DATE_ALIAS_WEIGHTS = (
    (120, ["order date", "sale date", "transaction date", "invoice date"]),
    (105, ["date", "order", "transaction", "invoice"]),
    (90, ["ship date", "delivery date", "created at", "updated at"]),
)

METRIC_ALIAS_WEIGHTS = (
    (150, ["sales", "revenue", "total sales", "total revenue", "net sales"]),
    (130, ["amount", "total", "income", "gmv"]),
    (115, ["profit", "earnings"]),
    (105, ["quantity", "qty", "units", "unit sold", "units sold"]),
    (90, ["price", "unit price", "cost"]),
)

GROUP_ALIAS_WEIGHTS = (
    (140, ["segment"]),
    (130, ["category"]),
    (120, ["sub-category", "sub category"]),
    (110, ["department", "brand"]),
    (100, ["region", "market"]),
    (90, ["state", "city"]),
    (80, ["product", "item", "store"]),
)

IDENTIFIER_PHRASES = {
    "row id",
    "order id",
    "customer id",
    "product id",
    "transaction id",
    "postal code",
    "zip code",
}

IDENTIFIER_TOKENS = {"id", "identifier", "code", "postal", "zip", "zipcode", "index"}

UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def format_display_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Not available"

    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):,.2f}"

    return str(value)


def _cache_file_path():
    os.makedirs(current_app.instance_path, exist_ok=True)
    return os.path.join(current_app.instance_path, "dataset_analysis.json")


def _normalize(value):
    cleaned = re.sub(r"[_\-]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", cleaned)


def _tokenize(value):
    return set(re.findall(r"[a-z0-9]+", _normalize(value)))


def _alias_score(column_name, weighted_aliases):
    normalized_name = _normalize(column_name)
    name_tokens = _tokenize(column_name)
    best_score = 0

    for weight, aliases in weighted_aliases:
        for alias in aliases:
            normalized_alias = _normalize(alias)
            alias_tokens = _tokenize(alias)

            if normalized_name == normalized_alias:
                best_score = max(best_score, weight + 20)
            elif normalized_alias in normalized_name or normalized_name in normalized_alias:
                best_score = max(best_score, weight + 10)
            elif alias_tokens and alias_tokens.issubset(name_tokens):
                best_score = max(best_score, weight)

    return best_score


def _is_identifier_column(column_name, series):
    normalized_name = _normalize(column_name)
    tokens = _tokenize(column_name)

    if normalized_name in IDENTIFIER_PHRASES:
        return True
    if IDENTIFIER_TOKENS.intersection(tokens):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)
    if unique_ratio > 0.95 and pd.api.types.is_numeric_dtype(non_null):
        return True

    return False


def _coerce_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_dates(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    sample = series.dropna()
    if sample.empty:
        return pd.to_datetime(series, errors="coerce")

    parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def _detect_date_column(dataframe):
    best_column = None
    best_score = -1

    for column in dataframe.columns:
        series = dataframe[column]
        alias_score = _alias_score(column, DATE_ALIAS_WEIGHTS)
        if pd.api.types.is_numeric_dtype(series) and alias_score == 0:
            continue

        parsed_dates = _parse_dates(series)
        non_null_original = series.dropna()
        if non_null_original.empty:
            continue

        parse_ratio = parsed_dates.notna().sum() / max(len(non_null_original), 1)
        if parse_ratio < 0.5:
            continue

        score = alias_score + int(parse_ratio * 100)
        if score > best_score:
            best_score = score
            best_column = column

    return best_column


def _detect_metric_column(dataframe, excluded_columns=None):
    excluded_columns = set(excluded_columns or [])
    best_column = None
    best_score = -1

    for column in dataframe.columns:
        if column in excluded_columns:
            continue

        series = dataframe[column]
        numeric_series = _coerce_numeric(series)
        valid_values = numeric_series.dropna()

        if valid_values.empty:
            continue
        if valid_values.notna().sum() / max(len(series.dropna()), 1) < 0.6:
            continue
        if _is_identifier_column(column, numeric_series):
            continue

        variability_score = 10 if valid_values.nunique() > 1 else -20
        positive_score = 8 if valid_values.sum() > 0 else 0
        alias_score = _alias_score(column, METRIC_ALIAS_WEIGHTS)
        uniqueness_penalty = -25 if (valid_values.nunique() / max(len(valid_values), 1)) > 0.98 else 0

        score = alias_score + variability_score + positive_score + uniqueness_penalty
        if score > best_score:
            best_score = score
            best_column = column

    return best_column


def _detect_grouping_column(dataframe, excluded_columns=None):
    excluded_columns = set(excluded_columns or [])
    best_column = None
    best_score = -1

    for column in dataframe.columns:
        if column in excluded_columns:
            continue

        series = dataframe[column]
        if _is_identifier_column(column, series):
            continue

        if pd.api.types.is_numeric_dtype(series):
            continue

        non_null = series.dropna().astype(str).str.strip()
        if non_null.empty:
            continue

        unique_count = non_null.nunique(dropna=True)
        if unique_count < 2:
            continue
        if unique_count > min(60, max(10, len(non_null) // 2)):
            continue

        alias_score = _alias_score(column, GROUP_ALIAS_WEIGHTS)
        readability_bonus = 20 if unique_count <= 12 else 10 if unique_count <= 25 else 0
        score = alias_score + readability_bonus

        if score > best_score:
            best_score = score
            best_column = column

    return best_column


def _build_trend_chart(dataframe, date_column, metric_column):
    empty_chart = {"title": "Dataset Trend", "labels": [], "values": [], "dataset_label": ""}
    if not date_column:
        return empty_chart

    working = dataframe.copy()
    working[date_column] = _parse_dates(working[date_column])
    working.dropna(subset=[date_column], inplace=True)
    if working.empty:
        return empty_chart

    if metric_column:
        working[metric_column] = _coerce_numeric(working[metric_column])
        working.dropna(subset=[metric_column], inplace=True)
        if working.empty:
            return empty_chart
        value_column = metric_column
        dataset_label = metric_column
    else:
        value_column = "__record_count__"
        working[value_column] = 1
        dataset_label = "Record Count"

    date_range_days = (working[date_column].max() - working[date_column].min()).days if len(working) > 1 else 0
    if date_range_days > 90 or working[date_column].nunique() > 45:
        frequency = "M"
        label_format = "%b %Y"
    else:
        frequency = "D"
        label_format = "%d %b"

    grouped = (
        working.set_index(date_column)
        .resample(frequency)[value_column]
        .sum()
        .tail(12)
    )

    return {
        "title": f"{dataset_label} Trend",
        "labels": [index.strftime(label_format) for index in grouped.index],
        "values": [round(float(value), 2) for value in grouped.tolist()],
        "dataset_label": dataset_label,
    }


def _build_grouped_chart(dataframe, group_column, metric_column, top_n, title, chart_label):
    empty_chart = {"title": title, "labels": [], "values": [], "dataset_label": chart_label}
    if not group_column:
        return empty_chart

    working = dataframe.copy()
    working[group_column] = working[group_column].astype(str).str.strip()
    working = working[working[group_column] != ""]
    if working.empty:
        return empty_chart

    if metric_column:
        working[metric_column] = _coerce_numeric(working[metric_column])
        working.dropna(subset=[metric_column], inplace=True)
        grouped = working.groupby(group_column)[metric_column].sum().sort_values(ascending=False).head(top_n)
        dataset_label = metric_column
    else:
        grouped = working[group_column].value_counts().head(top_n)
        dataset_label = "Record Count"

    if grouped.empty:
        return empty_chart

    return {
        "title": title,
        "labels": [str(index) for index in grouped.index.tolist()],
        "values": [round(float(value), 2) for value in grouped.tolist()],
        "dataset_label": dataset_label,
    }


def _build_distribution_chart(dataframe, metric_column):
    empty_chart = {"title": "Distribution", "labels": [], "values": [], "dataset_label": ""}
    if not metric_column:
        return empty_chart

    numeric_series = _coerce_numeric(dataframe[metric_column]).dropna()
    if numeric_series.empty:
        return empty_chart

    if numeric_series.nunique() <= 1:
        return {
            "title": f"{metric_column} Distribution",
            "labels": [format_display_value(numeric_series.iloc[0])],
            "values": [int(len(numeric_series))],
            "dataset_label": "Frequency",
        }

    bucket_count = min(8, max(4, numeric_series.nunique()))
    bins = pd.cut(numeric_series, bins=bucket_count, duplicates="drop")
    distribution = bins.value_counts(sort=False)

    return {
        "title": f"{metric_column} Distribution",
        "labels": [str(index) for index in distribution.index.tolist()],
        "values": [int(value) for value in distribution.tolist()],
        "dataset_label": "Frequency",
    }


def _build_summary_cards(dataframe, metric_column):
    summary_cards = [
        {"label": "Rows", "value": format_display_value(len(dataframe))},
        {"label": "Columns", "value": format_display_value(len(dataframe.columns))},
        {"label": "Missing Cells", "value": format_display_value(int(dataframe.isna().sum().sum()))},
    ]

    if metric_column:
        total_metric = _coerce_numeric(dataframe[metric_column]).dropna().sum()
        summary_cards.insert(0, {"label": f"Total {metric_column}", "value": format_display_value(total_metric)})
    else:
        summary_cards.insert(0, {"label": "Records", "value": format_display_value(len(dataframe))})

    return summary_cards


def _calculate_top_group(dataframe, group_column, metric_column):
    if not group_column:
        return "Not available"

    working = dataframe.copy()
    working[group_column] = working[group_column].astype(str).str.strip()
    working = working[working[group_column] != ""]
    if working.empty:
        return "Not available"

    if metric_column:
        working[metric_column] = _coerce_numeric(working[metric_column])
        working.dropna(subset=[metric_column], inplace=True)
        if working.empty:
            return "Not available"
        grouped = working.groupby(group_column)[metric_column].sum().sort_values(ascending=False)
    else:
        grouped = working[group_column].value_counts()

    if grouped.empty:
        return "Not available"

    return str(grouped.index[0])


def _build_analysis_note(date_column, metric_column, group_column):
    parts = []
    if date_column:
        parts.append(f"Using {date_column} for the time trend.")
    else:
        parts.append("No strong date column was detected.")

    if metric_column:
        parts.append(f"Using {metric_column} as the main numeric metric.")
    else:
        parts.append("No strong business metric was detected, so the dashboard falls back to record counts.")

    if group_column:
        parts.append(f"Using {group_column} for grouped breakdowns.")
    else:
        parts.append("No suitable grouping column was detected.")

    return " ".join(parts)


def _serialize_payload(payload):
    return json.loads(json.dumps(payload, default=str))


def analyze_uploaded_dataset(dataframe, dataset_name):
    working = dataframe.copy()
    working.columns = [str(column).strip() for column in working.columns]

    date_column = _detect_date_column(working)
    metric_column = _detect_metric_column(working, excluded_columns={date_column} if date_column else set())
    grouping_column = _detect_grouping_column(
        working,
        excluded_columns={value for value in [date_column, metric_column] if value},
    )

    payload = {
        "analysis_version": ANALYSIS_CACHE_VERSION,
        "dataset_name": dataset_name,
        "summary_cards": _build_summary_cards(working, metric_column),
        "charts": {
            "trend": _build_trend_chart(working, date_column, metric_column),
            "breakdown": _build_grouped_chart(
                working,
                grouping_column,
                metric_column,
                top_n=10,
                title="Grouped Breakdown",
                chart_label=metric_column or "Record Count",
            ),
            "composition": _build_grouped_chart(
                working,
                grouping_column,
                metric_column,
                top_n=6,
                title="Composition",
                chart_label=metric_column or "Record Count",
            ),
            "distribution": _build_distribution_chart(working, metric_column),
        },
        "insights": {
            "analysis_note": _build_analysis_note(date_column, metric_column, grouping_column),
            "date_column": date_column or "Not detected",
            "metric_column": metric_column or "Not detected",
            "category_column": grouping_column or "Not detected",
            "top_segment": _calculate_top_group(working, grouping_column, metric_column),
        },
    }

    return _serialize_payload(payload)


def save_dataset_analysis(payload):
    payload = dict(payload)
    payload["analysis_version"] = ANALYSIS_CACHE_VERSION
    cache_path = _cache_file_path()

    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(_serialize_payload(payload), cache_file, indent=2)


def _load_dataframe_from_file(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".csv":
        read_attempts = (
            {"encoding": "utf-8", "low_memory": False},
            {"encoding": "utf-8-sig", "low_memory": False},
            {"encoding": "latin1", "low_memory": False},
            {"encoding": "cp1252", "low_memory": False},
        )

        for options in read_attempts:
            try:
                return pd.read_csv(file_path, **options)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(file_path, low_memory=False)

    return pd.read_excel(file_path)


def _latest_upload_file() -> Optional[str]:
    upload_folder = current_app.config.get("UPLOAD_FOLDER")
    if not upload_folder or not os.path.isdir(upload_folder):
        return None

    files = []
    for file_name in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, file_name)
        extension = os.path.splitext(file_name)[1].lower()
        if os.path.isfile(file_path) and extension in UPLOAD_EXTENSIONS:
            files.append(file_path)

    if not files:
        return None

    return max(files, key=os.path.getmtime)


def _rebuild_analysis_from_latest_upload():
    latest_file = _latest_upload_file()
    if not latest_file:
        return None

    try:
        dataframe = _load_dataframe_from_file(latest_file)
        payload = analyze_uploaded_dataset(dataframe, os.path.basename(latest_file))
        save_dataset_analysis(payload)
        return payload
    except Exception:
        return None


def load_dataset_analysis():
    cache_path = _cache_file_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                payload = json.load(cache_file)

            if payload.get("analysis_version") == ANALYSIS_CACHE_VERSION:
                return payload
        except Exception:
            pass

    rebuilt_payload = _rebuild_analysis_from_latest_upload()
    if rebuilt_payload is not None:
        return rebuilt_payload

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                return json.load(cache_file)
        except Exception:
            return None

    return None
