# Retail Intelligence Dashboard

Retail Intelligence Dashboard is a Flask-based analytics app for uploading sales datasets, exploring them visually, and training a simple revenue forecasting model from imported retail records.

It supports two workflows:

- Dynamic dataset analysis for any supported CSV or Excel upload
- Retail-specific forecasting when the uploaded file matches the expected sales schema

## Features

- User registration and login with role-based access
- Automatic first-user admin bootstrap
- Admin-only dataset upload, model retraining, and user management
- Manager access to dashboard, reports, CSV export, and PDF export
- Dynamic chart generation for uploaded datasets, even when the file is not retail-specific
- Retail data import into the app database when required columns are detected
- Automatic next-month revenue forecasting using the best of Linear Regression and Random Forest
- Live daily, weekly, and monthly sales charts that refresh every 30 seconds
- Downloadable CSV and PDF reports

## How the app works

When a user uploads a file, the app first analyzes it generically:

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

For very large uploads, the app keeps dynamic analysis enabled but skips retail database import and model training once the file exceeds `5,000` rows.

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

Frontend:

- Jinja templates
- Bootstrap 5
- Chart.js

Storage:

- SQLite by default (`sqlite:///retail.db`)

## Project structure

```text
.
|-- app/
|   |-- __init__.py
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

### 3. Run the application

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

To stop the app, press `Ctrl+C` in the terminal.

If PowerShell blocks script activation, run this once in the same terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Default behavior

- The first registered user is assigned the `admin` role automatically
- Later registered users are created as `manager`
- Legacy `user` or `staff` roles are normalized to `manager`
- Upload and model directories are created automatically on startup
- Dataset analysis is cached in `instance/dataset_analysis.json`

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

- The README previously described PostgreSQL, dark mode, and other features that are not implemented in the current codebase. This version reflects the repository as it exists now.
- `app/forms.py` is currently empty and not required for the active workflow.
