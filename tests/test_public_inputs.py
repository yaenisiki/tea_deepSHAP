"""Small checks for the public input boundary and development-only freeze rule."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TEA_DATA_PATH", str(ROOT / "examples" / "synthetic_tea.csv"))
import lwt_data
from freeze_lwt import select_parameters
from lwt_models import MODEL_NAMES
import run_lwt_temporal


class PublicInputs(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("TEA_EXPECTED_SHA256", None)
        self.data = pd.read_csv(ROOT / "examples" / "synthetic_tea.csv")
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def load(self, data):
        source = self.directory / "input.csv"
        data.to_csv(source, index=False, encoding="utf-8")
        with patch.object(lwt_data, "DATA_PATH", source), patch.dict(os.environ, {}, clear=False):
            return lwt_data.load_dataset()

    def test_example_is_generated_independently(self):
        spec = importlib.util.spec_from_file_location("synthetic_generator", ROOT / "examples" / "generate_synthetic.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pd.testing.assert_frame_equal(self.data, module.generate())

    def test_extra_identifiers_are_dropped(self):
        self.data["original_identifier"] = "PRIVATE_TEST_IDENTIFIER"
        frame, digest = self.load(self.data)
        self.assertNotIn("original_identifier", frame)
        self.assertTrue(frame.analysis_id.is_unique)
        self.assertEqual(len(digest), 64)
        self.assertTrue((frame.loc[frame["品级"].eq("特级"), "label"] == 1).all())

    def test_invalid_inputs_are_rejected(self):
        for value in [np.nan, np.inf, "invalid"]:
            invalid = self.data.copy()
            invalid["GA"] = invalid["GA"].astype(object)
            invalid.loc[0, "GA"] = value
            with self.subTest(value=str(value)), self.assertRaises(ValueError):
                self.load(invalid)
        with self.assertRaises(ValueError):
            self.load(self.data.iloc[:-1])

    def test_input_digest_cannot_be_silently_changed(self):
        with patch.dict(os.environ, {"TEA_EXPECTED_SHA256": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.load(self.data)

    def test_absent_private_input_has_clear_error(self):
        with patch.object(lwt_data, "DATA_PATH", self.directory / "absent.xlsx"):
            with self.assertRaisesRegex(FileNotFoundError, "Private study data are not included"):
                lwt_data.load_dataset()

    def test_freeze_uses_inner_scores_instead_of_outer_scores(self):
        records = []
        for model in MODEL_NAMES:
            for index in range(25):
                parameter = {"model__C": 0.1 if index < 12 else (1.0 if index < 24 else 10.0)}
                records.append({"model": model, "repeat": index // 5 + 1, "fold": index % 5 + 1,
                    "best_params": json.dumps(parameter), "inner_best_auc": 0.9 if 12 <= index < 24 else 0.6,
                    "ROC_AUC": 1.0 if index < 12 else 0.1})
        folds = pd.DataFrame(records)
        chosen = select_parameters(folds)
        self.assertTrue(all(value == {"model__C": 1.0} for value in chosen.values()))
        folds["ROC_AUC"] = 1 - folds["ROC_AUC"]
        self.assertEqual(chosen, select_parameters(folds))
        with self.assertRaises(ValueError):
            select_parameters(folds.iloc[:-1])

    def test_optional_audit_does_not_assert_provenance_is_verified(self):
        with patch.object(run_lwt_temporal, "AUDIT_PATH", self.directory / "absent.csv"):
            self.assertEqual(run_lwt_temporal._flagged_rows(self.data), set())
        wrong = self.directory / "wrong.csv"
        pd.DataFrame({"unrelated_column": [1]}).to_csv(wrong, index=False)
        with patch.object(run_lwt_temporal, "AUDIT_PATH", wrong):
            with self.assertRaises(ValueError):
                run_lwt_temporal._flagged_rows(self.data)


if __name__ == "__main__":
    unittest.main()
