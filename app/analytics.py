import json
import os
import re
from typing import Optional

import pandas as pd
from flask import current_app

ANALYSIS_CACHE_VERSION = 4

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

DATE_PARSE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%d-%m-%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
)

RETAIL_COLUMN_ALIASES = {
    "date": ["transaction_date", "transaction date", "sale_date", "sale date", "order_date", "order date", "date"],
    "sales": ["sales_amount", "sales amount", "revenue", "sales", "amount", "total_sales", "total sales"],
    "quantity": ["quantity", "qty", "units", "units sold"],
    "unit_price": ["unit_price", "unit price", "price", "rate", "selling price"],
    "discount": ["discount_pct", "discount pct", "discount_percent", "discount percent", "discount", "discount rate"],
    "category": ["category", "product category", "department"],
    "product": ["product_name", "product name", "product", "item", "item name"],
    "segment": ["customer_segment", "customer segment", "segment"],
    "age_group": ["customer_age_group", "customer age group", "age_group", "age group", "age"],
    "region": ["region", "state", "city", "market"],
    "channel": ["sales_channel", "sales channel", "channel"],
}


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


def _find_column(columns, aliases):
    ranked_matches = []
    for column in columns:
        normalized_column = _normalize(column)
        column_tokens = _tokenize(column)

        for alias in aliases:
            normalized_alias = _normalize(alias)
            alias_tokens = _tokenize(alias)

            score = 0
            if normalized_column == normalized_alias:
                score = 100
            elif normalized_alias in normalized_column or normalized_column in normalized_alias:
                score = 80
            elif alias_tokens and alias_tokens.issubset(column_tokens):
                score = 60

            if score:
                ranked_matches.append((score, len(str(column)), column))

    if not ranked_matches:
        return None

    ranked_matches.sort(reverse=True)
    return ranked_matches[0][2]


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

    if pd.api.types.is_numeric_dtype(sample):
        return pd.to_datetime(series, errors="coerce")

    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned == "")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    for date_format in DATE_PARSE_FORMATS:
        remaining = parsed.isna() & cleaned.notna()
        if not remaining.any():
            break

        formatted = pd.to_datetime(cleaned[remaining], format=date_format, errors="coerce")
        parsed.loc[remaining] = formatted

    remaining = parsed.isna() & cleaned.notna()
    if remaining.any():
        try:
            mixed = pd.to_datetime(cleaned[remaining], format="mixed", errors="coerce")
            parsed.loc[remaining] = mixed
        except (TypeError, ValueError):
            pass

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
        frequency = "ME"
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


def _build_named_chart(title, dataset_label, labels, values, description=""):
    return {
        "title": title,
        "dataset_label": dataset_label,
        "labels": labels,
        "values": values,
        "description": description,
    }


def _build_insight_card(label, value, helper_text=""):
    return {
        "label": label,
        "value": value,
        "helper_text": helper_text,
    }


def _safe_percentage(numerator, denominator):
    if not denominator:
        return 0.0
    return (numerator / denominator) * 100


def _build_business_metric_card(label, value, helper_text=""):
    return {
        "label": label,
        "value": value,
        "helper_text": helper_text,
    }


def _build_uploaded_sales_performance(
    working,
    sales_column,
    quantity_column=None,
    unit_price_column=None,
    discount_column=None,
    category_column=None,
    channel_column=None,
    region_column=None,
):
    performance = {
        "title": "Sales Performance",
        "subtitle": "Calculated from price, quantity, discount, and final sales amount where those columns are available.",
        "note": "Profit metrics such as Gross Profit, EBITDA, and Net Profit need cost and expense data.",
        "cards": [],
        "charts": {},
    }

    if working.empty or not sales_column:
        return performance

    working = working.copy()
    working[sales_column] = _coerce_numeric(working[sales_column]).fillna(0)
    net_revenue = float(working[sales_column].sum())
    gross_sales = None
    total_units = None

    if quantity_column:
        working[quantity_column] = _coerce_numeric(working[quantity_column])
        total_units = float(working[quantity_column].dropna().sum())

    if quantity_column and unit_price_column:
        working[unit_price_column] = _coerce_numeric(working[unit_price_column])
        working["__gross_sales__"] = (working[quantity_column].fillna(0) * working[unit_price_column].fillna(0))
        gross_sales = float(working["__gross_sales__"].sum())
    elif discount_column:
        working[discount_column] = _coerce_numeric(working[discount_column])
        discount_rate = working[discount_column].clip(lower=0, upper=99.99) / 100
        working["__gross_sales__"] = working[sales_column] / (1 - discount_rate)
        gross_sales = float(working["__gross_sales__"].replace([float("inf"), -float("inf")], 0).fillna(0).sum())

    if gross_sales is not None:
        working["__discount_amount__"] = (working["__gross_sales__"] - working[sales_column]).clip(lower=0)
        discount_amount = float(working["__discount_amount__"].sum())
        discount_impact = _safe_percentage(discount_amount, gross_sales)
    else:
        discount_amount = None
        discount_impact = None

    average_selling_price = net_revenue / total_units if total_units else None
    average_order_value = net_revenue / len(working) if len(working) else 0

    if gross_sales is not None:
        performance["cards"].append(
            _build_business_metric_card("Gross Sales", format_display_value(gross_sales), "Before discounts")
        )
    performance["cards"].append(
        _build_business_metric_card("Net Revenue", format_display_value(net_revenue), "After discounts")
    )
    if discount_amount is not None:
        performance["cards"].append(
            _build_business_metric_card(
                "Discount Given",
                format_display_value(discount_amount),
                f"{discount_impact:.2f}% of gross sales",
            )
        )
    if total_units is not None:
        performance["cards"].append(
            _build_business_metric_card("Units Sold", format_display_value(total_units), "Total quantity moved")
        )
    if average_selling_price is not None:
        performance["cards"].append(
            _build_business_metric_card("Avg Selling Price", format_display_value(average_selling_price), "Net revenue per unit")
        )
    performance["cards"].append(
        _build_business_metric_card("Avg Order Value", format_display_value(average_order_value), "Net revenue per transaction")
    )

    graph_column = category_column or channel_column or region_column
    graph_label = (
        "Category"
        if graph_column == category_column
        else "Sales Channel"
        if graph_column == channel_column
        else "Region"
    )
    if gross_sales is not None and graph_column:
        sales_groups = (
            working.groupby(graph_column)
            .agg(
                gross_sales=("__gross_sales__", "sum"),
                net_revenue=(sales_column, "sum"),
            )
            .sort_values("net_revenue", ascending=False)
            .head(8)
        )
        labels = [str(index) for index in sales_groups.index.tolist()]
        performance["charts"]["gross_sales"] = _build_named_chart(
            f"Gross Sales by {graph_label}",
            "Gross Sales",
            labels,
            [round(float(value), 2) for value in sales_groups["gross_sales"].tolist()],
            f"Groups quantity multiplied by unit price across {graph_column}.",
        )
        performance["charts"]["net_sales"] = _build_named_chart(
            f"Net Sales by {graph_label}",
            "Net Sales",
            labels,
            [round(float(value), 2) for value in sales_groups["net_revenue"].tolist()],
            f"Groups final sales amount after discounts across {graph_column}.",
        )

    group_column = channel_column or region_column or category_column
    if group_column:
        grouped_revenue = (
            working.groupby(group_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )
        group_label = "Sales Channel" if group_column == channel_column else "Region" if group_column == region_column else "Category"
        performance["charts"]["revenue_mix"] = _build_named_chart(
            f"Net Revenue by {group_label}",
            "Net Revenue",
            [str(index) for index in grouped_revenue.index.tolist()],
            [round(float(value), 2) for value in grouped_revenue.tolist()],
            f"Compares final sales amount across {group_column}.",
        )

    return performance


def _build_uploaded_retail_payload(dataframe, dataset_name):
    columns = dataframe.columns.tolist()
    date_column = _find_column(columns, RETAIL_COLUMN_ALIASES["date"])
    sales_column = _find_column(columns, RETAIL_COLUMN_ALIASES["sales"])

    if not date_column or not sales_column:
        return None

    working = dataframe.copy()
    working[date_column] = _parse_dates(working[date_column])
    working[sales_column] = _coerce_numeric(working[sales_column])
    working.dropna(subset=[date_column, sales_column], inplace=True)
    if working.empty:
        return None

    quantity_column = _find_column(columns, RETAIL_COLUMN_ALIASES["quantity"])
    unit_price_column = _find_column(columns, RETAIL_COLUMN_ALIASES["unit_price"])
    discount_column = _find_column(columns, RETAIL_COLUMN_ALIASES["discount"])
    category_column = _find_column(columns, RETAIL_COLUMN_ALIASES["category"])
    product_column = _find_column(columns, RETAIL_COLUMN_ALIASES["product"])
    segment_column = _find_column(columns, RETAIL_COLUMN_ALIASES["segment"])
    age_group_column = _find_column(columns, RETAIL_COLUMN_ALIASES["age_group"])
    region_column = _find_column(columns, RETAIL_COLUMN_ALIASES["region"])
    channel_column = _find_column(columns, RETAIL_COLUMN_ALIASES["channel"])

    total_sales = float(working[sales_column].sum())
    transaction_count = int(len(working))
    average_order_value = total_sales / transaction_count if transaction_count else 0
    total_units = None
    if quantity_column:
        working[quantity_column] = _coerce_numeric(working[quantity_column])
        total_units = int(working[quantity_column].dropna().sum())

    date_range_days = (working[date_column].max() - working[date_column].min()).days if len(working) > 1 else 0
    trend_frequency = "ME" if date_range_days > 90 else "D"
    trend_label_format = "%b %Y" if trend_frequency == "ME" else "%d %b"
    sales_trend = (
        working.set_index(date_column)
        .resample(trend_frequency)[sales_column]
        .sum()
        .tail(12)
    )

    if category_column:
        category_sales = (
            working.groupby(category_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )
        breakdown_title = "Sales by Product Category"
        breakdown_description = f"Groups total {sales_column} by {category_column} to show which categories drive revenue."
    elif region_column:
        category_sales = (
            working.groupby(region_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )
        breakdown_title = "Sales by Region"
        breakdown_description = f"Groups total {sales_column} by {region_column} to show where sales come from."
    else:
        category_sales = pd.Series(dtype="float64")
        breakdown_title = "Sales Breakdown"
        breakdown_description = "No category or region column was detected for this breakdown."

    if product_column:
        top_products = (
            working.groupby(product_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )
        distribution_title = "Top Products by Revenue"
        distribution_label = "Revenue"
        distribution_description = f"Ranks products by total {sales_column}, so the tallest bars are the biggest revenue contributors."
    elif quantity_column and (category_column or segment_column or region_column):
        top_products = (
            working.groupby(category_column or segment_column or region_column)[quantity_column]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )
        distribution_title = "Top Groups by Units Sold"
        distribution_label = "Units Sold"
        distribution_description = f"Ranks groups by total {quantity_column}."
    else:
        top_products = pd.Series(dtype="float64")
        distribution_title = "Top Products"
        distribution_label = "Revenue"
        distribution_description = "No product or quantity column was detected."

    if age_group_column:
        composition = (
            working.groupby(age_group_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(6)
        )
        composition_title = "Customer Age Group Sales Share"
        composition_description = f"Compares total {sales_column} across {age_group_column} values to show which age groups drive sales."
        composition_source = age_group_column
    elif segment_column:
        composition = (
            working.groupby(segment_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(6)
        )
        composition_title = "Customer Segment Sales Share"
        composition_description = f"Compares total {sales_column} across {segment_column} values like Loyal, New, VIP, and Returning."
        composition_source = segment_column
    elif channel_column:
        composition = (
            working.groupby(channel_column)[sales_column]
            .sum()
            .sort_values(ascending=False)
            .head(6)
        )
        composition_title = "Sales Channel Share"
        composition_description = f"Compares total {sales_column} by {channel_column}, such as Online and In-Store."
        composition_source = channel_column
    else:
        composition = pd.Series(dtype="float64")
        composition_title = "Sales Share"
        composition_description = "No customer segment or sales channel column was detected."
        composition_source = None

    top_segment = str(category_sales.index[0]) if not category_sales.empty else "Not available"
    trend_basis = "monthly" if trend_frequency == "ME" else "daily"
    best_month = sales_trend.idxmax().strftime(trend_label_format) if not sales_trend.empty else "Not available"
    best_month_value = float(sales_trend.max()) if not sales_trend.empty else None
    best_product = str(top_products.index[0]) if not top_products.empty else "Not available"
    best_product_value = float(top_products.iloc[0]) if not top_products.empty else None
    top_age_group = str(composition.index[0]) if not composition.empty else "Not available"
    top_age_group_value = float(composition.iloc[0]) if not composition.empty else None
    insight_cards = [
        _build_insight_card(
            "Top Category",
            top_segment,
            f"{format_display_value(float(category_sales.iloc[0]))} sales" if not category_sales.empty else "",
        ),
        _build_insight_card(
            "Best Product",
            best_product,
            f"{format_display_value(best_product_value)} sales" if best_product_value is not None else "",
        ),
        _build_insight_card(
            "Best Sales Month",
            best_month,
            f"{format_display_value(best_month_value)} sales" if best_month_value is not None else "",
        ),
        _build_insight_card(
            "Top Age Group",
            top_age_group,
            f"{format_display_value(top_age_group_value)} sales" if top_age_group_value is not None else "",
        ),
    ]

    return _serialize_payload(
        {
            "analysis_version": ANALYSIS_CACHE_VERSION,
            "analysis_source": "uploaded_retail_dataset",
            "dataset_name": dataset_name,
            "summary_cards": [
                {"label": "Total Sales", "value": format_display_value(total_sales)},
                {"label": "Transactions", "value": format_display_value(transaction_count)},
                {"label": "Units Sold", "value": format_display_value(total_units) if total_units is not None else "Not detected"},
                {"label": "Average Order Value", "value": format_display_value(average_order_value)},
            ],
            "business_metrics": _build_uploaded_sales_performance(
                working,
                sales_column,
                quantity_column=quantity_column,
                unit_price_column=unit_price_column,
                discount_column=discount_column,
                category_column=category_column,
                channel_column=channel_column,
                region_column=region_column,
            ),
            "insight_cards": insight_cards,
            "charts": {
                "trend": _build_named_chart(
                    "Sales Trend Over Time",
                    "Sales Amount",
                    [index.strftime(trend_label_format) for index in sales_trend.index],
                    [round(float(value), 2) for value in sales_trend.tolist()],
                    f"Sums {sales_column} by {trend_basis} periods using {date_column}.",
                ),
                "breakdown": _build_named_chart(
                    breakdown_title,
                    "Sales Amount",
                    [str(index) for index in category_sales.index.tolist()],
                    [round(float(value), 2) for value in category_sales.tolist()],
                    breakdown_description,
                ),
                "composition": _build_named_chart(
                    composition_title,
                    "Sales Amount",
                    [str(index) for index in composition.index.tolist()],
                    [round(float(value), 2) for value in composition.tolist()],
                    composition_description,
                ),
                "distribution": _build_named_chart(
                    distribution_title,
                    distribution_label,
                    [str(index) for index in top_products.index.tolist()],
                    [round(float(value), 2) for value in top_products.tolist()],
                    distribution_description,
                ),
            },
            "insights": {
                "analysis_note": (
                    f"Retail analysis is based on {sales_column} as sales, {date_column} as transaction date, "
                    f"{category_column or region_column or 'detected groups'} for breakdowns, "
                    f"and {composition_source or 'available customer groups'} for share analysis."
                ),
                "date_column": date_column,
                "metric_column": sales_column,
                "category_column": category_column or region_column or "Not detected",
                "top_segment": top_segment,
            },
            "retail_compatible": False,
        }
    )


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

    retail_payload = _build_uploaded_retail_payload(working, dataset_name)
    if retail_payload:
        return retail_payload

    date_column = _detect_date_column(working)
    metric_column = _detect_metric_column(working, excluded_columns={date_column} if date_column else set())
    grouping_column = _detect_grouping_column(
        working,
        excluded_columns={value for value in [date_column, metric_column] if value},
    )

    payload = {
        "analysis_version": ANALYSIS_CACHE_VERSION,
        "analysis_source": "uploaded_dataset",
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


def load_dataset_analysis():
    cache_path = _cache_file_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                payload = json.load(cache_file)

            has_current_shape = (
                payload.get("analysis_source") != "uploaded_retail_dataset"
                or bool(payload.get("business_metrics"))
            )
            if payload.get("analysis_version") == ANALYSIS_CACHE_VERSION and has_current_shape:
                return payload
        except Exception:
            pass

    return None
