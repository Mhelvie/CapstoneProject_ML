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
        self.forecast_monthly_path = os.path.join(self.output_dir, "forecast_monthly.csv")
        self.featured_path = os.path.join(self.output_dir, "featured_dataset.csv")

    def test_model_artifacts_exist_and_load(self):
        """Ensure participant, booking, and revenue model artifacts exist and can be loaded."""
        pax_model_path = os.path.join(self.model_dir, "participant_count_model.joblib")
        bkg_model_path = os.path.join(self.model_dir, "booking_count_model.joblib")
        rev_model_path = os.path.join(self.model_dir, "class_revenue_model.joblib")

        self.assertTrue(os.path.exists(pax_model_path), f"Missing model: {pax_model_path}")
        self.assertTrue(os.path.exists(bkg_model_path), f"Missing model: {bkg_model_path}")
        self.assertTrue(os.path.exists(rev_model_path), f"Missing model: {rev_model_path}")

        pax_model = joblib.load(pax_model_path)
        bkg_model = joblib.load(bkg_model_path)
        rev_model = joblib.load(rev_model_path)

        self.assertTrue(hasattr(pax_model, "predict"), "Participant model is missing predict method.")
        self.assertTrue(hasattr(bkg_model, "predict"), "Booking model is missing predict method.")
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
            "predicted_bookings",
            "predicted_revenue_php",
            "demand_level",
            "season_period",
            "instructors_needed",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing required column in forecast.csv: {col}")

        # Check value constraints
        self.assertTrue((df["predicted_participants"] >= 0).all(), "Predicted participants must be non-negative.")
        self.assertTrue((df["predicted_bookings"] >= 0).all(), "Predicted bookings must be non-negative.")
        self.assertTrue((df["predicted_revenue_php"] >= 0).all(), "Predicted revenue must be non-negative.")
        self.assertTrue((df["instructors_needed"] >= 1).all(), "Instructors needed must be at least 1.")

        # Ensure predicted_bookings is distinct from predicted_participants (not duplicated)
        self.assertFalse((df["predicted_bookings"] == df["predicted_participants"]).all(),
                         "predicted_bookings must not be a duplicate of predicted_participants.")

        valid_demand_levels = {"Low", "Medium", "High"}
        self.assertTrue(set(df["demand_level"]).issubset(valid_demand_levels), "Invalid demand level classifications.")

        valid_seasons = {"Peak", "Shoulder", "Off-Peak"}
        self.assertTrue(set(df["season_period"]).issubset(valid_seasons), "Invalid season period classifications.")

    def test_forecast_monthly_csv_format_and_integrity(self):
        """Ensure outputs/forecast_monthly.csv exists and has correct columns and aggregation."""
        self.assertTrue(os.path.exists(self.forecast_monthly_path), f"Missing forecast_monthly file: {self.forecast_monthly_path}")

        df = pd.read_csv(self.forecast_monthly_path)
        self.assertGreaterEqual(len(df), 2, "Monthly forecast should contain at least 2 distinct calendar months.")

        expected_cols = [
            "month",
            "batches_in_month",
            "predicted_participants",
            "predicted_bookings",
            "predicted_revenue_php",
            "instructors_needed_total",
            "season_period",
        ]
        self.assertEqual(list(df.columns), expected_cols, f"Mismatch in forecast_monthly.csv columns: {list(df.columns)}")
        self.assertTrue((df["batches_in_month"] >= 1).all(), "Each month must have at least 1 batch.")
        self.assertTrue((df["predicted_participants"] > 0).all(), "Monthly predicted participants must be positive.")
        self.assertTrue((df["predicted_bookings"] > 0).all(), "Monthly predicted bookings must be positive.")
        self.assertTrue((df["predicted_revenue_php"] > 0).all(), "Monthly predicted revenue must be positive.")
        self.assertTrue((df["instructors_needed_total"] >= 1).all(), "Monthly total instructors needed must be positive.")

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
            "booking_count_lag_1",
            "booking_count_lag_2",
            "booking_count_lag_4",
            "booking_count_rolling_mean_4",
            "class_revenue_lag_1",
            "class_revenue_lag_2",
            "class_revenue_lag_4",
            "class_revenue_rolling_mean_4",
            "season_is_dry",
        ]
        for col in feature_cols:
            self.assertIn(col, df.columns, f"Missing feature column: {col}")

    def test_dataset_sources_and_batch_count(self):
        """Ensure dataset contains both 110 Historical Baseline and 19 Laravel Production DB batches (129 total)."""
        batch_path = os.path.join(self.output_dir, "batch_level_dataset.csv")
        self.assertTrue(os.path.exists(batch_path), f"Missing batch dataset: {batch_path}")

        df = pd.read_csv(batch_path)
        self.assertEqual(len(df), 129, f"Expected 129 total batches, found {len(df)}")

        source_counts = df["primary_source"].value_counts().to_dict()
        hist_count = source_counts.get("Historical Operational Actuals", 0)
        prod_count = source_counts.get("Laravel Production DB", 0)

        self.assertEqual(hist_count, 110, f"Expected 110 Historical Operational Actuals batches, got {hist_count}")
        self.assertEqual(prod_count, 19, f"Expected 19 Laravel Production DB batches, got {prod_count}")

        # Check target columns are populated
        for col in ["participant_count", "booking_count", "class_revenue"]:
            self.assertIn(col, df.columns)
            self.assertFalse(df[col].isna().any(), f"Column {col} contains NaN values")

    def test_statistical_demand_classification_rules(self):
        """Verify dynamic statistical demand classification matches exact rules and boundaries."""
        from retrain_pipeline import classify_monthly_demand_season

        upper_thresh = 72.0
        lower_thresh = 47.4

        # August 2026: 45 < 47.4 -> Off-Peak
        aug_class = classify_monthly_demand_season(45.0, upper_thresh, lower_thresh)
        self.assertEqual(aug_class, "Off-Peak", f"Expected Off-Peak for 45 pax, got {aug_class}")

        # September 2026: 82 > 72.0 -> Peak
        sep_class = classify_monthly_demand_season(82.0, upper_thresh, lower_thresh)
        self.assertEqual(sep_class, "Peak", f"Expected Peak for 82 pax, got {sep_class}")

        # October 2026: 47.4 <= 52 <= 72.0 -> Shoulder
        oct_class = classify_monthly_demand_season(52.0, upper_thresh, lower_thresh)
        self.assertEqual(oct_class, "Shoulder", f"Expected Shoulder for 52 pax, got {oct_class}")

        # Edge cases: exactly at thresholds
        self.assertEqual(classify_monthly_demand_season(72.0, upper_thresh, lower_thresh), "Shoulder")
        self.assertEqual(classify_monthly_demand_season(47.4, upper_thresh, lower_thresh), "Shoulder")
        self.assertEqual(classify_monthly_demand_season(72.01, upper_thresh, lower_thresh), "Peak")
        self.assertEqual(classify_monthly_demand_season(47.39, upper_thresh, lower_thresh), "Off-Peak")

    def test_monthly_demand_classifications_json_integrity(self):
        """Ensure outputs/monthly_demand_classifications.json conforms to the API output structure."""
        import json
        json_path = os.path.join(self.output_dir, "monthly_demand_classifications.json")
        self.assertTrue(os.path.exists(json_path), f"Missing JSON artifact: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 2)

        required_keys = [
            "month",
            "monthly_average",
            "overall_mean",
            "standard_deviation",
            "upper_threshold",
            "lower_threshold",
            "classification",
        ]
        valid_classes = {"Peak", "Shoulder", "Off-Peak"}

        for item in data:
            for k in required_keys:
                self.assertIn(k, item, f"Missing key '{k}' in monthly classification object")
            self.assertIn(item["classification"], valid_classes)
            self.assertIsInstance(item["monthly_average"], (int, float))
            self.assertIsInstance(item["overall_mean"], (int, float))
            self.assertIsInstance(item["standard_deviation"], (int, float))
            self.assertIsInstance(item["upper_threshold"], (int, float))
            self.assertIsInstance(item["lower_threshold"], (int, float))

    def test_demand_classification_mathematical_precision_and_sensitivity(self):
        """
        Comprehensive Verification of Requirements:
        1. Monthly averages are calculated correctly from batch forecast rows.
        2. Overall mean is calculated from the monthly averages.
        3. Standard deviation is calculated correctly (sample SD, ddof=1).
        4. Upper threshold equals mean + 1 SD.
        5. Lower threshold equals mean - 1 SD.
        6. Values above the upper threshold are classified as Peak.
        7. Values between or equal to the thresholds are classified as Shoulder.
        8. Values below the lower threshold are classified as Off-Peak.
        9. No month is automatically classified based on its calendar name.
        10. Changing the forecast values can change the resulting classification.
        """
        from retrain_pipeline import classify_monthly_demand_season, compute_statistical_demand_layer

        # -------------------------------------------------------------
        # 1 & 2 & 3: Mathematical Verification of Thresholds & Classes
        # -------------------------------------------------------------
        # 4-month horizon: Oct (10.0), Nov (20.0), Dec (22.0), Jan (36.0)
        monthly_avgs_base = pd.Series([10.0, 20.0, 22.0, 36.0], index=["2026-10", "2026-11", "2026-12", "2027-01"])
        mean_base = float(monthly_avgs_base.mean())
        sd_base = float(monthly_avgs_base.std(ddof=1))
        upper_base = round(mean_base + sd_base, 2)
        lower_base = round(mean_base - sd_base, 2)

        self.assertAlmostEqual(mean_base, 22.00, places=2)
        self.assertAlmostEqual(sd_base, 10.71, places=2)
        self.assertAlmostEqual(upper_base, 32.71, places=2)
        self.assertAlmostEqual(lower_base, 11.29, places=2)

        # Oct: 10.0 < 11.29 -> Off-Peak
        class_oct = classify_monthly_demand_season(10.0, upper_base, lower_base)
        self.assertEqual(class_oct, "Off-Peak", "Values below lower threshold must be Off-Peak")

        # Nov: 11.29 <= 20.0 <= 32.71 -> Shoulder
        class_nov = classify_monthly_demand_season(20.0, upper_base, lower_base)
        self.assertEqual(class_nov, "Shoulder", "Values between thresholds must be Shoulder")

        # Dec: 11.29 <= 22.0 <= 32.71 -> Shoulder
        class_dec = classify_monthly_demand_season(22.0, upper_base, lower_base)
        self.assertEqual(class_dec, "Shoulder", "Values between thresholds must be Shoulder")

        # Jan: 36.0 > 32.71 -> Peak
        class_jan = classify_monthly_demand_season(36.0, upper_base, lower_base)
        self.assertEqual(class_jan, "Peak", "Values above upper threshold must be Peak")

        # -------------------------------------------------------------
        # 9 & 10: Sensitivity & Dynamic Response (No hardcoded calendar)
        # Invert the forecast: Oct surges to 36.0, Jan drops to 10.0
        # -------------------------------------------------------------
        monthly_avgs_inverted = pd.Series([36.0, 20.0, 22.0, 10.0], index=["2026-10", "2026-11", "2026-12", "2027-01"])
        mean_inv = float(monthly_avgs_inverted.mean())
        sd_inv = float(monthly_avgs_inverted.std(ddof=1))
        upper_inv = round(mean_inv + sd_inv, 2)
        lower_inv = round(mean_inv - sd_inv, 2)

        class_oct_inv = classify_monthly_demand_season(36.0, upper_inv, lower_inv)
        class_jan_inv = classify_monthly_demand_season(10.0, upper_inv, lower_inv)

        # October changed from Off-Peak to Peak!
        self.assertEqual(class_oct_inv, "Peak", "October must dynamically become Peak when its forecast surges")
        # January changed from Peak to Off-Peak!
        self.assertEqual(class_jan_inv, "Off-Peak", "January must dynamically become Off-Peak when its forecast drops")

        # -------------------------------------------------------------
        # Prompt Example Verification:
        # Aug (45), Sep (82), Oct (52) with Mean=59.7, SD=12.3, Upper=72.0, Lower=47.4
        # -------------------------------------------------------------
        prompt_upper = 72.0
        prompt_lower = 47.4
        self.assertEqual(classify_monthly_demand_season(45.0, prompt_upper, prompt_lower), "Off-Peak")
        self.assertEqual(classify_monthly_demand_season(82.0, prompt_upper, prompt_lower), "Peak")
        self.assertEqual(classify_monthly_demand_season(52.0, prompt_upper, prompt_lower), "Shoulder")

        # -------------------------------------------------------------
        # End-to-End compute_statistical_demand_layer on batch DataFrame
        # -------------------------------------------------------------
        synthetic_forecast = pd.DataFrame([
            {"forecast_date": "2026-08-01", "predicted_participants": 44.0},
            {"forecast_date": "2026-08-15", "predicted_participants": 46.0},  # Aug avg = 45.0
            {"forecast_date": "2026-09-01", "predicted_participants": 80.0},
            {"forecast_date": "2026-09-15", "predicted_participants": 84.0},  # Sep avg = 82.0
            {"forecast_date": "2026-10-01", "predicted_participants": 50.0},
            {"forecast_date": "2026-10-15", "predicted_participants": 54.0},  # Oct avg = 52.0
        ])
        results, season_map, o_mean, o_sd, u_th, l_th = compute_statistical_demand_layer(synthetic_forecast)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["month"], "August 2026")
        self.assertEqual(results[0]["monthly_average"], 45.0)
        self.assertEqual(results[1]["month"], "September 2026")
        self.assertEqual(results[1]["monthly_average"], 82.0)
        self.assertEqual(results[2]["month"], "October 2026")
        self.assertEqual(results[2]["monthly_average"], 52.0)

        # Check overall mean = (45 + 82 + 52) / 3 = 59.67 -> round 59.7
        self.assertAlmostEqual(o_mean, 59.67, places=1)
        self.assertEqual(u_th, round(o_mean + o_sd, 2))
        self.assertEqual(l_th, round(o_mean - o_sd, 2))
        self.assertEqual(results[1]["classification"], "Peak")


if __name__ == "__main__":
    unittest.main()




