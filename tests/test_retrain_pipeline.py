"""
Unit and smoke tests for continuous retraining pipeline and artifacts.
"""

import os
import unittest
import joblib
import pandas as pd


class TestRetrainPipeline(unittest.TestCase):
    def setUp(self):
        self.output_dir = "outputs"
        self.model_dir = os.path.join(self.output_dir, "models")
        self.forecast_path = os.path.join(self.output_dir, "forecast.csv")
        self.featured_path = os.path.join(self.output_dir, "featured_dataset.csv")

    def test_model_artifacts_exist_and_load(self):
        """Ensure participant and revenue model artifacts exist and can be loaded."""
        pax_model_path = os.path.join(self.model_dir, "participant_count_model.joblib")
        rev_model_path = os.path.join(self.model_dir, "class_revenue_model.joblib")

        self.assertTrue(os.path.exists(pax_model_path), f"Missing model: {pax_model_path}")
        self.assertTrue(os.path.exists(rev_model_path), f"Missing model: {rev_model_path}")

        pax_model = joblib.load(pax_model_path)
        rev_model = joblib.load(rev_model_path)

        self.assertTrue(hasattr(pax_model, "predict"), "Participant model is missing predict method.")
        self.assertTrue(hasattr(rev_model, "predict"), "Revenue model is missing predict method.")

    def test_forecast_csv_format_and_integrity(self):
        """Ensure outputs/forecast.csv is generated, non-empty, and conforms to required schema."""
        self.assertTrue(os.path.exists(self.forecast_path), f"Missing forecast file: {self.forecast_path}")

        df = pd.read_csv(self.forecast_path)
        self.assertGreaterEqual(len(df), 12, "Forecast should contain at least 12 weekly horizon points.")

        expected_cols = [
            "forecast_date",
            "days_ahead",
            "predicted_participants",
            "predicted_revenue_php",
            "demand_level",
            "season_period",
            "instructors_needed",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing required column in forecast.csv: {col}")

        # Check value constraints
        self.assertTrue((df["predicted_participants"] >= 0).all(), "Predicted participants must be non-negative.")
        self.assertTrue((df["predicted_revenue_php"] >= 0).all(), "Predicted revenue must be non-negative.")
        self.assertTrue((df["instructors_needed"] >= 1).all(), "Instructors needed must be at least 1.")

        valid_demand_levels = {"Low", "Medium", "High"}
        self.assertTrue(set(df["demand_level"]).issubset(valid_demand_levels), "Invalid demand level classifications.")

        valid_seasons = {"Peak", "Shoulder", "Off-Peak"}
        self.assertTrue(set(df["season_period"]).issubset(valid_seasons), "Invalid season period classifications.")

    def test_featured_dataset_integrity(self):
        """Ensure outputs/featured_dataset.csv contains all engineered feature columns."""
        self.assertTrue(os.path.exists(self.featured_path), f"Missing featured dataset: {self.featured_path}")
        df = pd.read_csv(self.featured_path)
        self.assertGreater(len(df), 50, "Featured dataset should have sufficient historical depth.")

        feature_cols = [
            "day_of_week",
            "week_of_year",
            "month",
            "day_of_year",
            "is_weekend",
            "avg_lead_time",
            "median_lead_time",
            "participant_count_lag_1",
            "participant_count_lag_2",
            "participant_count_lag_4",
            "participant_count_rolling_mean_4",
            "class_revenue_lag_1",
            "class_revenue_lag_2",
            "class_revenue_lag_4",
            "class_revenue_rolling_mean_4",
            "season_is_dry",
        ]
        for col in feature_cols:
            self.assertIn(col, df.columns, f"Missing feature column: {col}")


if __name__ == "__main__":
    unittest.main()
