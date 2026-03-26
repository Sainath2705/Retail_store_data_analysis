import os
import tempfile
import unittest
from datetime import datetime

from app import create_app, db
from app.ml_model import predict_next_month, sync_model_with_sales_data
from app.models import Product, Sale, Store, User


class RetailAppFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(
            {
                "TESTING": True,
                "SECRET_KEY": "test-secret",
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{os.path.join(self.temp_dir.name, 'test.db')}",
                "UPLOAD_FOLDER": os.path.join(self.temp_dir.name, "uploads"),
                "MODEL_FOLDER": os.path.join(self.temp_dir.name, "models"),
            }
        )
        self.client = self.app.test_client()

        with self.app.app_context():
            db.drop_all()
            db.create_all()
            self._seed_users()
            self._seed_sales()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _seed_users(self):
        admin = User(username="admin", email="admin@example.com", role="admin")
        admin.set_password("password")

        manager = User(username="manager", email="manager@example.com", role="manager")
        manager.set_password("password")

        # Keep one legacy role to verify backward compatibility for older rows.
        staff = User(username="staff", email="staff@example.com", role="user")
        staff.set_password("password")

        db.session.add_all([admin, manager, staff])
        db.session.commit()

    def _seed_sales(self):
        store = Store(name="Central Store", city="Hyderabad", state="Telangana")
        product = Product(name="Laptop", category="Electronics", price=50000)
        db.session.add_all([store, product])
        db.session.flush()

        sales_dates = [
            datetime(2024, 1, 10),
            datetime(2024, 2, 12),
            datetime(2024, 3, 14),
            datetime(2024, 4, 16),
            datetime(2024, 5, 18),
            datetime(2024, 6, 20),
        ]
        revenues = [1200, 1450, 1700, 1900, 2200, 2600]

        for index, sale_date in enumerate(sales_dates):
            sale = Sale(
                store_id=store.id,
                product_id=product.id,
                quantity=index + 1,
                revenue=revenues[index],
                sale_date=sale_date,
            )
            db.session.add(sale)

        db.session.commit()

    def _login(self, username):
        return self.client.post(
            "/login",
            data={"username": username, "password": "password"},
            follow_redirects=True,
        )

    def _logout(self):
        return self.client.get("/logout", follow_redirects=True)

    def test_dashboard_page_and_chart_api_load(self):
        self._login("staff")

        page_response = self.client.get("/")
        self.assertEqual(page_response.status_code, 200)
        self.assertIn(b"Live Sales Charts", page_response.data)

        api_response = self.client.get("/api/dashboard/sales-summary")
        self.assertEqual(api_response.status_code, 200)

        payload = api_response.get_json()
        self.assertIn("daily", payload)
        self.assertIn("weekly", payload)
        self.assertIn("monthly", payload)
        self.assertTrue(isinstance(payload["daily"]["labels"], list))

    def test_retraining_updates_prediction_when_sales_change(self):
        with self.app.app_context():
            sync_model_with_sales_data(force=True)
            baseline_prediction = predict_next_month()

            store = Store.query.first()
            product = Product.query.first()
            new_sale = Sale(
                store_id=store.id,
                product_id=product.id,
                quantity=10,
                revenue=10000,
                sale_date=datetime(2024, 7, 25),
            )
            db.session.add(new_sale)
            db.session.commit()

        self._login("admin")
        response = self.client.post("/train-model", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model retrained successfully", response.data)

        with self.app.app_context():
            updated_prediction = predict_next_month()

        self.assertIsNotNone(baseline_prediction)
        self.assertIsNotNone(updated_prediction)
        self.assertNotEqual(baseline_prediction, updated_prediction)

    def test_csv_export_works_for_manager(self):
        self._login("manager")

        response = self.client.get("/reports/export/csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers.get("Content-Type", ""))
        self.assertIn(b"Sale ID", response.data)
        self.assertIn(b"Revenue", response.data)

    def test_role_restrictions_are_enforced(self):
        self._login("staff")
        reports_response = self.client.get("/reports", follow_redirects=False)
        retrain_response = self.client.post("/train-model", follow_redirects=False)
        self.assertEqual(reports_response.status_code, 302)
        self.assertEqual(retrain_response.status_code, 302)

        self._logout()
        self._login("manager")
        reports_response = self.client.get("/reports")
        retrain_response = self.client.post("/train-model", follow_redirects=False)
        self.assertEqual(reports_response.status_code, 200)
        self.assertEqual(retrain_response.status_code, 302)

        self._logout()
        self._login("admin")
        reports_response = self.client.get("/reports")
        retrain_response = self.client.post("/train-model", follow_redirects=False)
        self.assertEqual(reports_response.status_code, 200)
        self.assertEqual(retrain_response.status_code, 302)


if __name__ == "__main__":
    unittest.main()
