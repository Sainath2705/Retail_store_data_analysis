import logging
from datetime import datetime

from flask import current_app

from app.models import Product, Sale, Store
from app import db
from sqlalchemy import func

logger = logging.getLogger(__name__)

_model_instance = None
_genai = None


def _get_genai():
    global _genai
    if _genai is False:
        return None
    if _genai is not None:
        return _genai

    try:
        import google.generativeai as genai_module
    except ModuleNotFoundError:
        logger.warning(
            "google-generativeai is not installed. AI insights will be unavailable."
        )
        _genai = False
        return None

    _genai = genai_module
    return _genai


def _get_model():
    """Initialize and cache the Gemini model."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    api_key = current_app.config.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set. AI insights will be unavailable.")
        return None

    genai = _get_genai()
    if genai is None:
        return None

    genai.configure(api_key=api_key)
    _model_instance = genai.GenerativeModel("gemini-2.0-flash")
    return _model_instance


def _build_sales_context(user_id=None):
    """Build a compact text summary of the sales data for Gemini context."""

    query = db.session.query(Sale)
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    
    total_sales = query.count() or 0
    if total_sales == 0:
        return None

    total_revenue = db.session.query(func.coalesce(func.sum(Sale.revenue), 0.0))
    if user_id is not None:
        total_revenue = total_revenue.filter_by(user_id=user_id)
    total_revenue = total_revenue.scalar()
    
    total_units_query = db.session.query(func.coalesce(func.sum(Sale.quantity), 0))
    if user_id is not None:
        total_units_query = total_units_query.filter_by(user_id=user_id)
    total_units = total_units_query.scalar()
    
    min_date_query = db.session.query(func.min(Sale.sale_date))
    if user_id is not None:
        min_date_query = min_date_query.filter_by(user_id=user_id)
    min_date = min_date_query.scalar()
    
    max_date_query = db.session.query(func.max(Sale.sale_date))
    if user_id is not None:
        max_date_query = max_date_query.filter_by(user_id=user_id)
    max_date = max_date_query.scalar()
    
    avg_revenue = round(total_revenue / total_sales, 2) if total_sales else 0

    # Top 5 categories by revenue
    top_categories_query = (
        db.session.query(Product.category, func.sum(Sale.revenue).label("rev"))
        .join(Sale, Sale.product_id == Product.id)
    )
    if user_id is not None:
        top_categories_query = top_categories_query.filter(Sale.user_id == user_id)
    top_categories = (
        top_categories_query
        .group_by(Product.category)
        .order_by(func.sum(Sale.revenue).desc())
        .limit(5)
        .all()
    )

    # Top 5 products by revenue
    top_products_query = (
        db.session.query(Product.name, func.sum(Sale.revenue).label("rev"))
        .join(Sale, Sale.product_id == Product.id)
    )
    if user_id is not None:
        top_products_query = top_products_query.filter(Sale.user_id == user_id)
    top_products = (
        top_products_query
        .group_by(Product.name)
        .order_by(func.sum(Sale.revenue).desc())
        .limit(5)
        .all()
    )

    # Top 5 stores by revenue
    top_stores_query = (
        db.session.query(Store.name, Store.city, func.sum(Sale.revenue).label("rev"))
        .join(Sale, Sale.store_id == Store.id)
    )
    if user_id is not None:
        top_stores_query = top_stores_query.filter(Sale.user_id == user_id)
    top_stores = (
        top_stores_query
        .group_by(Store.name, Store.city)
        .order_by(func.sum(Sale.revenue).desc())
        .limit(5)
        .all()
    )

    # Monthly revenue trend (last 6 months)
    monthly_trend_query = (
        db.session.query(
            func.strftime("%Y-%m", Sale.sale_date).label("month"),
            func.sum(Sale.revenue).label("rev"),
            func.sum(Sale.quantity).label("qty"),
        )
    )
    if user_id is not None:
        monthly_trend_query = monthly_trend_query.filter(Sale.user_id == user_id)
    monthly_trend = (
        monthly_trend_query
        .group_by(func.strftime("%Y-%m", Sale.sale_date))
        .order_by(func.strftime("%Y-%m", Sale.sale_date).desc())
        .limit(6)
        .all()
    )

    # Build the context string
    lines = [
        "=== RETAIL SALES DATA SUMMARY ===",
        f"Date range: {min_date} to {max_date}",
        f"Total sales records: {total_sales:,}",
        f"Total revenue: ₹{total_revenue:,.2f}",
        f"Total units sold: {total_units:,}",
        f"Average sale value: ₹{avg_revenue:,.2f}",
        "",
        "Top categories by revenue:",
    ]
    for cat, rev in top_categories:
        lines.append(f"  - {cat}: ₹{rev:,.2f}")

    lines.append("")
    lines.append("Top products by revenue:")
    for name, rev in top_products:
        lines.append(f"  - {name}: ₹{rev:,.2f}")

    lines.append("")
    lines.append("Top stores by revenue:")
    for name, city, rev in top_stores:
        lines.append(f"  - {name} ({city}): ₹{rev:,.2f}")

    lines.append("")
    lines.append("Monthly revenue trend (recent):")
    for month, rev, qty in reversed(monthly_trend):
        lines.append(f"  - {month}: ₹{rev:,.2f} ({qty:,} units)")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are a retail analytics AI assistant for a Retail Intelligence Dashboard.
You analyze sales data and provide actionable business insights.

Rules:
- Be concise and data-driven. Use numbers from the provided data.
- Format responses in clean markdown with bullet points.
- Focus on actionable insights, trends, and recommendations.
- If the user asks something unrelated to retail/sales, politely redirect.
- Use Indian Rupee (₹) for currency.
- Keep responses under 300 words unless the user asks for a detailed analysis.
"""


def generate_auto_summary(user_id=None):
    """Generate an automatic AI summary of the current sales data."""
    model = _get_model()
    if model is None:
        return None

    context = _build_sales_context(user_id=user_id)
    if context is None:
        return None

    prompt = f"""{SYSTEM_PROMPT}

Here is the current sales data:

{context}

Generate a brief executive summary (4-6 bullet points) covering:
1. Overall performance highlights
2. Top performing category/product and why it matters
3. Any concerning trends or drops
4. One actionable recommendation for the store manager
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as exc:
        logger.error("Gemini auto-summary failed: %s", exc)
        return None


def ask_question(user_question, user_id=None):
    """Answer a user's question about the sales data using Gemini."""
    model = _get_model()
    if model is None:
        return {"error": "AI insights are unavailable. GEMINI_API_KEY is not configured."}

    context = _build_sales_context(user_id=user_id)
    if context is None:
        return {"error": "No sales data available. Upload a dataset first."}

    prompt = f"""{SYSTEM_PROMPT}

Here is the current sales data:

{context}

User's question: {user_question}

Answer the question based on the data above. If the question cannot be answered from the data, say so clearly.
"""

    try:
        response = model.generate_content(prompt)
        return {"answer": response.text, "question": user_question}
    except Exception as exc:
        logger.error("Gemini Q&A failed: %s", exc)
        return {"error": f"AI request failed: {str(exc)}"}
