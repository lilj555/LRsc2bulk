import os
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from tcga_survival_analysis import parse_lr_pair_name, load_clinical, compute_lr_expression


class TestTcgASurvivalAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("/home/lilj/work/xenium/code/tests/tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # 清理临时文件
        for p in self.tmp_dir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass

    def test_parse_lr_pair_name(self):
        cases = {
            "LIG|REC": ("LIG", "REC"),
            "LIG_REC": ("LIG", "REC"),
            "LIG->REC": ("LIG", "REC"),
            "LIG-REC": ("LIG", "REC"),
            "LIG/REC": ("LIG", "REC"),
        }
        for k, v in cases.items():
            self.assertEqual(parse_lr_pair_name(k), v)
        self.assertIsNone(parse_lr_pair_name("INVALID"))

    def test_load_clinical_detection(self):
        df = pd.DataFrame({
            "bcr_patient_barcode": ["S1", "S2", "S3"],
            "days_to_death": [10, np.nan, 30],
            "days_to_last_follow_up": [np.nan, 40, np.nan],
            "vital_status": ["Dead", "Alive", "Dead"],
        })
        p = self.tmp_dir / "clin.csv"
        df.to_csv(p, index=False)
        clin = load_clinical(p)
        self.assertIn("duration", clin.columns)
        self.assertIn("event", clin.columns)
        self.assertEqual(clin.loc["S1", "event"], 1)
        self.assertEqual(clin.loc["S2", "event"], 0)

    def test_compute_lr_expression(self):
        expr = pd.DataFrame({
            "LIG": [1.0, 2.0, 3.0],
            "REC": [4.0, 5.0, 6.0],
            "OTHER": [7.0, 8.0, 9.0],
        }, index=["S1", "S2", "S3"])
        lr_pairs = [("LIG|REC", "LIG", "REC")]
        lr_expr = compute_lr_expression(expr, [("LIG|REC", "LIG", "REC")], method="geomean")
        self.assertIn("LIG|REC", lr_expr.columns)
        self.assertEqual(lr_expr.shape[0], 3)


if __name__ == "__main__":
    unittest.main()

