import csv
from io import BytesIO, StringIO

import pandas as pd

from app.models import Product, Sale, Store


def build_sales_dataframe(user_id=None):
    columns = [
        "sale_id",
        "store_name",
        "city",
        "state",
        "product_name",
        "category",
        "price",
        "quantity",
        "revenue",
        "sale_date",
    ]

    query = Sale.query
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    sales = query.order_by(Sale.sale_date.asc()).all()
    if not sales:
        return pd.DataFrame(columns=columns)

    product_map = {product.id: product for product in Product.query.all()}
    store_map = {store.id: store for store in Store.query.all()}

    rows = []
    for sale in sales:
        product = product_map.get(sale.product_id)
        store = store_map.get(sale.store_id)
        rows.append(
            {
                "sale_id": sale.id,
                "store_name": store.name if store else "Unknown Store",
                "city": store.city if store else "Unknown",
                "state": store.state if store else "Unknown",
                "product_name": product.name if product else "Unknown Product",
                "category": product.category if product else "General",
                "price": float(product.price) if product and product.price is not None else 0.0,
                "quantity": int(sale.quantity or 0),
                "revenue": float(sale.revenue or 0),
                "sale_date": sale.sale_date,
            }
        )

    dataframe = pd.DataFrame(rows, columns=columns)
    dataframe["sale_date"] = pd.to_datetime(dataframe["sale_date"], errors="coerce")
    dataframe.dropna(subset=["sale_date"], inplace=True)
    dataframe.sort_values("sale_date", inplace=True)
    return dataframe


def _build_series(dataframe, frequency, label_formatter, limit):
    if dataframe.empty:
        return {"labels": [], "values": []}

    grouped = (
        dataframe.set_index("sale_date")
        .resample(frequency)["revenue"]
        .sum()
        .tail(limit)
    )

    return {
        "labels": [label_formatter(index) for index in grouped.index],
        "values": [round(float(value), 2) for value in grouped.tolist()],
    }


def build_sales_chart_payload(dataframe=None, user_id=None):
    if dataframe is None:
        dataframe = build_sales_dataframe(user_id=user_id)

    daily = _build_series(dataframe, "D", lambda index: index.strftime("%d %b"), 14)
    daily.update({"title": "Daily Sales", "dataset_label": "Daily Revenue"})

    weekly = _build_series(
        dataframe,
        "W-MON",
        lambda index: f"Week of {index.strftime('%d %b')}",
        12,
    )
    weekly.update({"title": "Weekly Sales", "dataset_label": "Weekly Revenue"})

    monthly = _build_series(
        dataframe,
        "ME",
        lambda index: index.strftime("%b %Y"),
        12,
    )
    monthly.update({"title": "Monthly Sales", "dataset_label": "Monthly Revenue"})

    return {"daily": daily, "weekly": weekly, "monthly": monthly}


def build_sales_overview_cards(dataframe=None, user_id=None):
    if dataframe is None:
        dataframe = build_sales_dataframe(user_id=user_id)
    if dataframe.empty:
        return [
            {"label": "Sales Records", "value": "0"},
            {"label": "Total Revenue", "value": "0.00"},
            {"label": "Units Sold", "value": "0"},
            {"label": "Average Sale Value", "value": "0.00"},
        ]

    total_sales = int(len(dataframe))
    total_revenue = float(dataframe["revenue"].sum())
    total_units = int(dataframe["quantity"].sum())
    average_sale_value = total_revenue / total_sales if total_sales else 0.0

    return [
        {"label": "Sales Records", "value": f"{total_sales:,}"},
        {"label": "Total Revenue", "value": f"{total_revenue:,.2f}"},
        {"label": "Units Sold", "value": f"{total_units:,}"},
        {"label": "Average Sale Value", "value": f"{average_sale_value:,.2f}"},
    ]


def _format_number(value):
    if value is None:
        return "Not available"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):,.2f}"


def _business_metric_card(label, value, helper_text=""):
    return {
        "label": label,
        "value": value,
        "helper_text": helper_text,
    }


def build_sales_performance_payload(dataframe=None, user_id=None):
    if dataframe is None:
        dataframe = build_sales_dataframe(user_id=user_id)

    payload = {
        "title": "Sales Performance",
        "subtitle": "Calculated from stored product price, quantity, and final revenue.",
        "note": "Profit metrics such as Gross Profit, EBITDA, and Net Profit need cost and expense data.",
        "cards": [],
        "charts": {},
    }

    if dataframe.empty:
        return payload

    working = dataframe.copy()
    working["price"] = pd.to_numeric(working["price"], errors="coerce").fillna(0)
    working["quantity"] = pd.to_numeric(working["quantity"], errors="coerce").fillna(0)
    working["revenue"] = pd.to_numeric(working["revenue"], errors="coerce").fillna(0)
    working["gross_sales"] = working["price"] * working["quantity"]
    working["discount_amount"] = (working["gross_sales"] - working["revenue"]).clip(lower=0)

    gross_sales = float(working["gross_sales"].sum())
    net_revenue = float(working["revenue"].sum())
    discount_amount = float(working["discount_amount"].sum())
    total_units = float(working["quantity"].sum())
    average_selling_price = net_revenue / total_units if total_units else None
    average_order_value = net_revenue / len(working) if len(working) else 0
    discount_impact = (discount_amount / gross_sales * 100) if gross_sales else 0

    payload["cards"] = [
        _business_metric_card("Gross Sales", _format_number(gross_sales), "Before discounts"),
        _business_metric_card("Net Revenue", _format_number(net_revenue), "After discounts"),
        _business_metric_card("Discount Given", _format_number(discount_amount), f"{discount_impact:.2f}% of gross sales"),
        _business_metric_card("Units Sold", _format_number(total_units), "Total quantity moved"),
        _business_metric_card(
            "Avg Selling Price",
            _format_number(average_selling_price) if average_selling_price is not None else "Not available",
            "Net revenue per unit",
        ),
        _business_metric_card("Avg Order Value", _format_number(average_order_value), "Net revenue per transaction"),
    ]

    sales_by_category = (
        working.groupby("category")
        .agg(
            gross_sales=("gross_sales", "sum"),
            net_revenue=("revenue", "sum"),
        )
        .sort_values("net_revenue", ascending=False)
        .head(8)
    )
    revenue_by_store = (
        working.groupby("store_name")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
    )

    payload["charts"] = {
        "gross_sales": _build_named_chart(
            "Gross Sales by Category",
            "Gross Sales",
            [str(index) for index in sales_by_category.index.tolist()],
            [round(float(value), 2) for value in sales_by_category["gross_sales"].tolist()],
        ),
        "net_sales": _build_named_chart(
            "Net Sales by Category",
            "Net Sales",
            [str(index) for index in sales_by_category.index.tolist()],
            [round(float(value), 2) for value in sales_by_category["net_revenue"].tolist()],
        ),
        "revenue_mix": _build_named_chart(
            "Net Revenue by Store",
            "Net Revenue",
            [str(index) for index in revenue_by_store.index.tolist()],
            [round(float(value), 2) for value in revenue_by_store.tolist()],
        ),
    }

    return payload


def build_report_rows(limit=None, user_id=None):
    dataframe = build_sales_dataframe(user_id=user_id).sort_values("sale_date", ascending=False)
    if limit is not None:
        dataframe = dataframe.head(limit)

    rows = []
    for _, row in dataframe.iterrows():
        rows.append(
            {
                "Sale ID": int(row["sale_id"]),
                "Sale Date": row["sale_date"].strftime("%Y-%m-%d"),
                "Store": row["store_name"],
                "City": row["city"],
                "State": row["state"],
                "Product": row["product_name"],
                "Category": row["category"],
                "Quantity": int(row["quantity"]),
                "Revenue": round(float(row["revenue"]), 2),
            }
        )

    return rows


def build_report_summary(user_id=None):
    dataframe = build_sales_dataframe(user_id=user_id)
    total_sales = int(len(dataframe))
    total_revenue = float(dataframe["revenue"].sum()) if not dataframe.empty else 0.0
    total_units = int(dataframe["quantity"].sum()) if not dataframe.empty else 0
    average_revenue = total_revenue / total_sales if total_sales else 0.0

    return {
        "total_sales": total_sales,
        "total_sales_display": f"{total_sales:,}",
        "total_revenue": round(total_revenue, 2),
        "total_revenue_display": f"{total_revenue:,.2f}",
        "total_units": total_units,
        "total_units_display": f"{total_units:,}",
        "average_revenue": round(average_revenue, 2),
        "average_revenue_display": f"{average_revenue:,.2f}",
    }


def build_sales_csv_file(user_id=None):
    rows = build_report_rows(user_id=user_id)
    fieldnames = [
        "Sale ID",
        "Sale Date",
        "Store",
        "City",
        "State",
        "Product",
        "Category",
        "Quantity",
        "Revenue",
    ]

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    buffer = BytesIO(output.getvalue().encode("utf-8"))
    buffer.seek(0)
    return buffer


def _build_named_chart(title, dataset_label, labels, values):
    return {
        "title": title,
        "dataset_label": dataset_label,
        "labels": labels,
        "values": values,
    }


def build_retail_analysis_payload(dataset_name="Retail Sales Records", dataframe=None, user_id=None):
    if dataframe is None:
        dataframe = build_sales_dataframe(user_id=user_id)
    if dataframe.empty:
        return None

    monthly_revenue = (
        dataframe.set_index("sale_date")
        .resample("ME")["revenue"]
        .sum()
        .tail(12)
    )
    category_revenue = (
        dataframe.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
    )
    store_revenue = (
        dataframe.groupby("store_name")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
    )
    product_quantity = (
        dataframe.groupby("product_name")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
    )

    top_category = category_revenue.index[0] if not category_revenue.empty else "Not available"
    last_sale_date = dataframe["sale_date"].max()

    return {
        "analysis_source": "retail_database",
        "dataset_name": dataset_name,
        "summary_cards": [
            {"label": "Imported Sales Rows", "value": f"{len(dataframe):,}"},
            {"label": "Stores", "value": f"{dataframe['store_name'].nunique():,}"},
            {"label": "Products", "value": f"{dataframe['product_name'].nunique():,}"},
            {"label": "Categories", "value": f"{dataframe['category'].nunique():,}"},
        ],
        "charts": {
            "trend": _build_named_chart(
                "Monthly Revenue Trend",
                "Revenue",
                [index.strftime("%b %Y") for index in monthly_revenue.index],
                [round(float(value), 2) for value in monthly_revenue.tolist()],
            ),
            "breakdown": _build_named_chart(
                "Revenue by Category",
                "Revenue",
                [str(index) for index in category_revenue.index.tolist()],
                [round(float(value), 2) for value in category_revenue.tolist()],
            ),
            "composition": _build_named_chart(
                "Revenue Share by Store",
                "Revenue",
                [str(index) for index in store_revenue.index.tolist()],
                [round(float(value), 2) for value in store_revenue.tolist()],
            ),
            "distribution": _build_named_chart(
                "Top Products by Units Sold",
                "Units Sold",
                [str(index) for index in product_quantity.index.tolist()],
                [int(value) for value in product_quantity.tolist()],
            ),
        },
        "insights": {
            "analysis_note": "These retail insights are built from the imported sales records in your database.",
            "date_column": "Sale Date",
            "metric_column": "Revenue",
            "category_column": "Category",
            "top_segment": str(top_category),
            "top_segment_label": "Top Category",
            "last_sale_date": last_sale_date.strftime("%Y-%m-%d") if pd.notna(last_sale_date) else "Not available",
        },
    }
