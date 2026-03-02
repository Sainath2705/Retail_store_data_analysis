from app.models import Sale, Product
import pandas as pd


def get_dashboard_data():
    sales = Sale.query.all()

    if not sales:
        return None

    data = []

    for s in sales:
        product = Product.query.get(s.product_id)
        data.append({
            "product": product.name,
            "category": product.category,
            "quantity": s.quantity,
            "revenue": s.revenue,
            "date": s.sale_date
        })

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
        "monthly_trend": {str(k): v for k, v in monthly_trend.items()}
    }