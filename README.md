# Retail Intelligence Dashboard

Retail Intelligence Dashboard is a Flask-based analytics app for uploading sales datasets, exploring them visually, training a revenue forecasting model, and getting AI-powered business insights using Google Gemini.

It supports two workflows:

- Dynamic dataset analysis for any supported CSV or Excel upload
- Retail-specific forecasting when the uploaded file matches the expected sales schema

## Features

- User registration and login with role-based access
- Automatic first-user admin bootstrap
- Admin-only dataset upload, model retraining, and user management
- Manager access to dashboard, reports, CSV export, and PDF export
- Dynamic chart generation for uploaded datasets, even when the file is not retail-specific
- Retail-specific uploaded CSV analysis for large files without importing every row into the database
- Clear retail dashboard charts for sales trends, category performance, customer age groups, and top products
- Key Sales Insights cards for top category, best product, best sales month, and top customer age group
- Animated dashboard cards and Chart.js graphs with reduced-motion support
- Retail data import into the app database when required columns are detected
- Automatic next-month revenue forecasting using the best of Linear Regression and Random Forest
- Live daily, weekly, and monthly sales charts that refresh every 30 seconds
- Downloadable CSV and PDF reports
- **AI-powered insights** — auto-generated executive summaries and natural language Q&A about your sales data via Google Gemini

## How the app works

When a user uploads a file, the app first checks whether it looks like a retail sales dataset.

For retail-shaped uploaded files, the dashboard analyzes the CSV directly and shows business-focused cards and charts:

- Summary cards: total sales, transactions, units sold, and average order value
- Sales Trend Over Time: monthly sales based on `transaction_date` and `sales_amount`
- Sales by Product Category: total `sales_amount` grouped by `category`
- Customer Age Group Sales Share: total `sales_amount` grouped by `customer_age_group`
- Top Products by Revenue: products ranked by total `sales_amount`
- Key Sales Insights: top category, best product, best sales month, and top age group

The recommended uploaded retail dataset columns are:

- `transaction_id`
- `transaction_date`
- `customer_id`
- `customer_gender`
- `customer_age_group`
- `customer_segment`
- `product_id`
- `product_name`
- `category`
- `quantity`
- `unit_price`
- `discount_pct`
- `sales_amount`
- `payment_method`
- `sales_channel`
- `region`

For non-retail uploads, the app falls back to generic dataset analysis:

- Detects a likely date column
- Detects a likely numeric metric
- Detects a likely grouping column
- Builds summary cards and charts from those detected fields

If the file also looks like retail sales data, the app can import it into the database and enable forecasting. The essential fields are:

- `product_name`
- `quantity`
- `revenue`
- `sale_date`

The upload flow also tries to map common aliases such as `product`, `qty`, `sales`, `amount`, `date`, `store`, and `category`.

For very large uploads, the app keeps direct uploaded-dataset analysis enabled but skips retail database import once the file exceeds `5,000` rows. This keeps dashboards fast for large CSV files while still allowing visual analysis and uploaded-dataset forecasting.

## AI Insights (Gemini)

The dashboard includes AI-powered analytics via Google Gemini:

- **Auto Summary** (`GET /api/ai/summary`) — generates an executive summary with performance highlights, top categories, trends, and recommendations
- **Ask a Question** (`POST /api/ai/ask`) — lets users ask natural language questions about their sales data, e.g. *"Which product should I stock more?"* or *"Why did revenue drop in February?"*

The AI context is built directly from SQL aggregations (no full data loading), keeping it fast even with large datasets.

### Setup

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Add it to your `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Forecasting

Retail forecasting is based on monthly aggregated revenue from imported sales records.

- Candidate models: `LinearRegression` and `RandomForestRegressor`
- Selection rule: highest R2 score
- Output: next-month revenue prediction shown on the dashboard
- Training behavior: automatic retraining after a compatible import, or manual retraining from the dashboard

## Tech stack

Backend:

- Python
- Flask
- Flask-Login
- Flask-SQLAlchemy
- Pandas
- scikit-learn
- ReportLab
- Google Generative AI (Gemini)

Frontend:

- Jinja templates
- Bootstrap 5
- Chart.js
- CSS dashboard animations with `prefers-reduced-motion` support

Storage:

- SQLite by default (`sqlite:///retail.db`)

## Configuration

The app uses a `.env` file for secrets and configuration. Create one in the project root:

```env
SECRET_KEY=your-secret-key-here
GEMINI_API_KEY=your-gemini-api-key
```

The `.env` file is already in `.gitignore` and will never be committed.

## Project structure

```text
.
|-- app/
|   |-- __init__.py
|   |-- ai_insights.py
|   |-- analytics.py
|   |-- auth.py
|   |-- decorators.py
|   |-- ml_model.py
|   |-- models.py
|   |-- routes.py
|   |-- utils.py
|   `-- templates/
|-- tests/
|   `-- test_app_features.py
|-- .env
|-- sample_sales_data.csv
|-- config.py
|-- requirements.txt
|-- run.py
`-- README.md
```

## Getting started

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your-secret-key
GEMINI_API_KEY=your-gemini-api-key
```

### 4. Run the application

```bash
python run.py
```

The development server starts on `http://127.0.0.1:5001`.

## How to run the project

### Windows PowerShell

```powershell
cd "d:\Projects\da projects\retail store sales analysis"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

### macOS/Linux

```bash
cd /path/to/retail-store-sales-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

### After the server starts

1. Open `http://127.0.0.1:5001` in your browser.
2. Register a new account.
3. The first registered account becomes the `admin`.
4. Log in and upload `sample_sales_data.csv` if you want demo data quickly.
5. Open the dashboard to view charts, reports, and the forecast.
6. Use the AI insights API to get auto-generated summaries and ask questions about your data.

To stop the app, press `Ctrl+C` in the terminal.

If PowerShell blocks script activation, run this once in the same terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/` | Login | Dashboard with charts and overview |
| GET | `/api/dashboard/sales-summary` | Login | Sales chart data (JSON) |
| GET | `/api/ai/summary` | Login | AI-generated executive summary |
| POST | `/api/ai/ask` | Login | Ask a question about sales data |
| GET | `/reports` | Admin/Manager | Reports page |
| GET | `/reports/export/csv` | Admin/Manager | Download sales CSV |
| GET | `/download-report` | Admin/Manager | Download sales PDF |
| POST | `/upload` | Admin | Upload dataset |
| POST | `/train-model` | Admin | Retrain forecast model |
| GET/POST | `/admin/users` | Admin | User management |

## Default behavior

- The first registered user is assigned the `admin` role automatically
- Later registered users are created as `manager`
- Legacy `user` or `staff` roles are normalized to `manager`
- Upload and model directories are created automatically on startup
- Dataset analysis is cached in `instance/dataset_analysis.json`
- AI insights use SQL aggregations for context, not full data loading

## Sample dataset

Use [`sample_sales_data.csv`](sample_sales_data.csv) to try the retail import flow quickly. Its columns already match the expected retail schema:

- `store_name`
- `city`
- `state`
- `product_name`
- `category`
- `price`
- `quantity`
- `revenue`
- `sale_date`

## Running tests

```bash
python -m unittest discover -s tests
```

The current test suite covers:

- Dashboard and chart API loading
- Role restrictions for admin and manager actions
- CSV export
- Model retraining after sales data changes

## Notes

- Secrets are loaded from `.env` using `python-dotenv`. Never hardcode API keys in source code.
- Unused dependencies (`Flask-WTF`, `psycopg2-binary`, `plotly`, `joblib`, `email-validator`) have been removed from requirements.
- `app/forms.py` is currently empty and not required for the active workflow.

## Optimizations & Performance
- **Dashboard DB Queries:** Re-architected to fetch the main sales DataFrame exactly once per page load and pass it to downstream chart payload builders, avoiding multiple full-table scans.
- **Model Signature Check:** Refactored `get_sales_signature()` to use native SQL aggregations (`func.count`, `func.sum`) instead of loading all sales records into Python memory.
