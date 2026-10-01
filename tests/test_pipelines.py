"""Regression checks using synthetic samples and temporary artifacts only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ("kidney", "Nav", "Felix/kidney")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="medai-tests-")
        cls.work = Path(cls.temp.name)
        rng = np.random.default_rng(42)
        y = np.tile([0, 1], 30)
        cls.frame = pd.DataFrame(
            rng.normal(size=(len(y), 24)),
            columns=[f"feature_{idx:04d}" for idx in range(24)],
        )
        cls.frame["feature_0000"] += y * 1.5
        cls.frame["sample_id"] = [f"synthetic-{idx:03d}" for idx in range(len(y))]
        cls.frame["age"] = rng.uniform(30, 80, len(y))
        cls.frame["sex"] = rng.integers(1, 3, len(y))
        cls.frame["baseline_egfr_23"] = rng.uniform(20, 100, len(y))
        cls.frame["ati"] = y
        cls.data = cls.work / "train.csv"
        cls.frame.to_csv(cls.data, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_command(self, *args, env=None):
        result = subprocess.run(
            [str(arg) for arg in args],
            cwd=self.work,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def run_script(self, track, script, *args, env=None):
        return self.run_command(sys.executable, ROOT / track / script, *args, env=env)

    def assert_predictions(self, path, *, labeled):
        result = pd.read_csv(path)
        self.assertEqual(result["sample_id"].tolist(), self.frame["sample_id"].tolist())
        self.assertTrue(np.isfinite(result["prob_ati"]).all())
        self.assertTrue(result["prob_ati"].between(0, 1).all())
        self.assertTrue(result["pred_label"].isin([0, 1]).all())
        self.assertEqual("true_label" in result, labeled)
        return result

    def test_all_cli_help_without_checkpoints(self):
        for track in TRACKS:
            for script in ("evaluate.py", "train.py", "predict.py"):
                with self.subTest(track=track, script=script):
                    result = self.run_script(track, script, "--help")
                    self.assertIn("--data", result.stdout)

    def test_main_pipeline_and_parallel_reproducibility(self):
        results = []
        for workers, hash_seed in ((1, "11"), (2, "29")):
            out = self.work / f"main-results-{workers}"
            self.run_script(
                "kidney",
                "evaluate.py",
                "--data",
                self.data,
                "--out",
                out,
                "--models",
                "xgb_fs2",
                "--folds",
                "2",
                "--repeats",
                "1",
                "--workers",
                str(workers),
                "--xgb-n-jobs",
                "1",
                env={"PYTHONHASHSEED": hash_seed},
            )
            results.append(out)
        pd.testing.assert_frame_equal(
            pd.read_csv(results[0] / "base_oof_predictions_mean.csv"),
            pd.read_csv(results[1] / "base_oof_predictions_mean.csv"),
        )

        weights = self.work / "main-weights"
        self.run_script(
            "kidney",
            "train.py",
            "--data",
            self.data,
            "--results-dir",
            results[0],
            "--out",
            weights,
            "--xgb-n-jobs",
            "1",
        )
        output = self.work / "nested" / "main.csv"
        self.run_script(
            "kidney",
            "predict.py",
            "--data",
            self.data,
            "--model-dir",
            weights,
            "--out",
            output,
            "--include-base-probs",
        )
        expected = self.assert_predictions(output, labeled=True)
        self.assertIn("prob_xgb_fs2", expected)

        # Labels, extra columns, and input column order cannot affect inference.
        unlabeled = self.frame.drop(columns="ati").iloc[:, ::-1].copy()
        unlabeled["unused_column"] = "ignored"
        shuffled_data = self.work / "unlabeled.csv"
        unlabeled.to_csv(shuffled_data, index=False)
        wrapped_output = self.work / "wrapper" / "main.csv"
        self.run_command(
            "bash",
            ROOT / "kidney" / "predict.sh",
            shuffled_data,
            wrapped_output,
            "--model-dir",
            weights,
            env={"MEDAI_PYTHON": sys.executable},
        )
        actual = self.assert_predictions(wrapped_output, labeled=False)
        np.testing.assert_allclose(expected["prob_ati"], actual["prob_ati"])

    def test_nav_model_families_and_artifact_round_trip(self):
        results = self.work / "nav-results"
        weights = self.work / "nav-weights"
        self.run_script(
            "Nav",
            "evaluate.py",
            "--data",
            self.data,
            "--out",
            results,
            "--folds",
            "3",
            "--repeats",
            "1",
            "--xgb-n-jobs",
            "1",
        )
        metrics = pd.read_csv(results / "cv_results.csv")
        self.assertEqual(metrics["model"].nunique(), 5)
        self.run_script(
            "Nav",
            "train.py",
            "--data",
            self.data,
            "--recipe",
            results / "final_recipe.json",
            "--out",
            weights,
            "--xgb-n-jobs",
            "1",
        )
        output = self.work / "nested" / "nav.csv"
        self.run_command(
            "bash",
            ROOT / "Nav" / "predict.sh",
            self.data,
            output,
            "--model-dir",
            weights,
            env={"MEDAI_PYTHON": sys.executable},
        )
        self.assert_predictions(output, labeled=True)

    def test_nav_fallback_respects_custom_model_directory(self):
        # A custom directory with no ensemble config must load its own XGBoost.
        weights = self.work / "nav-fallback"
        weights.mkdir()
        features = ["feature_0000", "age"]
        model = XGBClassifier(n_estimators=3, max_depth=1, n_jobs=1, random_state=42)
        model.fit(self.frame[features], self.frame["ati"])
        model.save_model(weights / "xgboost_model.json")
        (weights / "feature_cols.json").write_text(json.dumps(features))
        output = self.work / "nested" / "fallback.csv"
        self.run_script(
            "Nav",
            "predict.py",
            "--data",
            self.data,
            "--model-dir",
            weights,
            "--out",
            output,
        )
        actual = self.assert_predictions(output, labeled=True)
        np.testing.assert_allclose(
            actual["prob_ati"], model.predict_proba(self.frame[features])[:, 1]
        )

    def test_fold_feature_selection_and_alignment(self):
        # Run each track in a separate process because its legacy module names overlap.
        code = """
import sys
import numpy as np
import pandas as pd
from preprocess import build_feature_frame, select_feature_columns

frame = pd.read_csv(sys.argv[1])
y = frame['ati'].to_numpy()
selected = select_feature_columns(frame.iloc[:40], y[:40], protein_top_k=3)
assert len(selected) == 6
assert 'ati' not in selected and 'sample_id' not in selected
expected = build_feature_frame(frame, selected)
changed = frame.iloc[:, ::-1].copy()
changed['ati'] = 1 - y
pd.testing.assert_frame_equal(build_feature_frame(changed, selected), expected)
missing = build_feature_frame(frame.drop(columns=selected[0]), selected)
assert missing[selected[0]].isna().all()
pd.testing.assert_frame_equal(missing[selected[1:]], expected[selected[1:]])
"""
        for track in ("Nav", "Felix/kidney"):
            with self.subTest(track=track):
                self.run_command(
                    sys.executable,
                    "-c",
                    code,
                    self.data,
                    env={"PYTHONPATH": str(ROOT / track)},
                )

    def test_tabpfn_prediction_artifact_without_checkpoint_download(self):
        # Exercise artifact loading, feature alignment, and the wrapper with a
        # lightweight sklearn classifier; actual TabPFN fitting needs its checkpoint.
        import joblib
        from sklearn.linear_model import LogisticRegression

        weights = self.work / "tabpfn-interface"
        weights.mkdir()
        features = ["feature_0000", "feature_0001", "age"]
        model = LogisticRegression(max_iter=500).fit(self.frame[features], self.frame["ati"])
        joblib.dump(model, weights / "tabpfn_model.joblib")
        (weights / "feature_cols.json").write_text(json.dumps(features))
        output = self.work / "nested" / "tabpfn-interface.csv"
        self.run_command(
            "bash",
            ROOT / "Felix/kidney" / "predict.sh",
            self.data,
            output,
            "--model-dir",
            weights,
            env={"MEDAI_PYTHON": sys.executable},
        )
        actual = self.assert_predictions(output, labeled=True)
        np.testing.assert_allclose(
            actual["prob_ati"], model.predict_proba(self.frame[features])[:, 1]
        )


if __name__ == "__main__":
    unittest.main()
