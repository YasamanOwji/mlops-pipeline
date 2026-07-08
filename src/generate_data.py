"""
pipeline.py
------------
MLOps pipeline with DVC, MLflow, deployment, and monitoring hooks.
Paths are resolved relative to the project root for robustness.
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Evidently imports
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationQualityPreset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MLOpsPipeline:
    """CI/CD pipeline for training, versioning, and deploying an ML model."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            # مسیر را نسبت به ریشه‌ی پروژه تنظیم کن
            root_dir = Path(__file__).resolve().parent.parent
            config_path = root_dir / "config" / "mlops_config.yaml"
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        self.client = MlflowClient()
        self.root_dir = Path(__file__).resolve().parent.parent

    # ------------------------------------------------------------------
    # 1. Data versioning
    # ------------------------------------------------------------------
    def version_data(self, data_path: str) -> str:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info("Adding %s to DVC tracking", data_path)
        subprocess.run(["dvc", "add", str(data_path)], check=True)

        remote_name = self.config["dvc"]["remote_name"]
        logger.info("Pushing data to DVC remote '%s'", remote_name)
        subprocess.run(["dvc", "push", "-r", remote_name], check=True)

        dvc_file = data_path.with_suffix(data_path.suffix + ".dvc")
        with open(dvc_file, "r", encoding="utf-8") as f:
            dvc_meta = yaml.safe_load(f)
        data_hash = dvc_meta["outs"][0]["md5"]

        logger.info("Data version (md5): %s", data_hash)
        return data_hash

    # ------------------------------------------------------------------
    # 2. Train & register
    # ------------------------------------------------------------------
    def train_and_register(self, data_path: str, data_version: str, model_params: Dict[str, Any]) -> str:
        df = pd.read_csv(data_path)
        target_col = self.config["data"]["target_column"]
        X = df.drop(columns=[target_col, "student_id"], errors="ignore")
        X = pd.get_dummies(X, drop_first=True)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        with mlflow.start_run() as run:
            mlflow.log_params(model_params)
            mlflow.log_param("data_version", data_version)

            model = self._train_model(X_train, y_train, model_params)
            metrics = self._evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(model, artifact_path="model")

            registered_name = self.config["mlflow"]["registered_model_name"]
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=registered_name,
            )

            # ذخیره داده‌های تست برای مانیتورینگ بعدی
            test_data_path = self.root_dir / "data" / "test_data.csv"
            test_data_path.parent.mkdir(exist_ok=True)
            test_df = X_test.copy()
            test_df[target_col] = y_test
            test_df.to_csv(test_data_path, index=False)
            mlflow.log_artifact(str(test_data_path), artifact_path="test_data")

            logger.info("Run %s complete. Metrics: %s", run.info.run_id, metrics)
            return run.info.run_id

    def _train_model(self, X_train, y_train, model_params: Dict[str, Any]):
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        return model

    def _evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        return {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }

    # ------------------------------------------------------------------
    # 3. Deploy
    # ------------------------------------------------------------------
    def deploy_model(self, run_id: str, environment: str = "staging") -> str:
        registered_name = self.config["mlflow"]["registered_model_name"]

        versions = self.client.search_model_versions(f"name='{registered_name}'")
        matching = [v for v in versions if v.run_id == run_id]
        if not matching:
            raise ValueError(f"No registered model version found for run_id={run_id}")
        version = matching[0].version

        model_uri = f"runs:/{run_id}/model"

        if environment == "production":
            logger.info("Promoting version %s of '%s' to Production", version, registered_name)
            self.client.transition_model_version_stage(
                name=registered_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            target_dir = self.root_dir / self.config["deployment"]["production_dir"]
        else:
            logger.info("Deploying version %s of '%s' to staging", version, registered_name)
            self.client.transition_model_version_stage(
                name=registered_name,
                version=version,
                stage="Staging",
            )
            target_dir = self.root_dir / self.config["deployment"]["staging_dir"]

        self._materialize_model(model_uri, target_dir)
        return str(target_dir)

    def _materialize_model(self, model_uri: str, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(mlflow.sklearn.load_model(model_uri), str(target_dir / "model"))
        with open(target_dir / "version.txt", "w") as f:
            f.write(model_uri)
        logger.info("Model artifact written to %s", target_dir)

    # ------------------------------------------------------------------
    # 4. Monitoring (Data Drift & Quality)
    # ------------------------------------------------------------------
    def run_monitoring(self, production_model_path: str, new_data_path: str, reference_data_path: str = None) -> dict:
        if reference_data_path is None:
            reference_data_path = self.root_dir / "data" / "test_data.csv"

        ref_df = pd.read_csv(reference_data_path)
        new_df = pd.read_csv(new_data_path)

        target_col = self.config["data"]["target_column"]

        # Data Drift report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=ref_df, current_data=new_df)
        drift_json = drift_report.as_dict()
        drift_score = drift_json['metrics'][0]['result']['drift_score']

        # Quality report (اگر target در new_data موجود باشد)
        quality_metrics = {}
        if target_col in new_df.columns:
            model_path = Path(production_model_path) / "model"
            if model_path.exists():
                model = mlflow.sklearn.load_model(str(model_path))
                X_new = new_df.drop(columns=[target_col, "student_id"], errors="ignore")
                X_new = pd.get_dummies(X_new, drop_first=True)
                # تطابق ستون‌ها با داده‌های مرجع (برای جلوگیری از خطا)
                common_cols = list(set(ref_df.columns) & set(X_new.columns))
                if common_cols:
                    X_new = X_new[common_cols]
                    y_true = new_df[target_col]
                    y_pred = model.predict(X_new)
                    quality_metrics = {
                        "accuracy": accuracy_score(y_true, y_pred),
                        "f1_score": f1_score(y_true, y_pred),
                    }

        # ذخیره گزارش
        report_dir = self.root_dir / self.config["monitoring"]["report_dir"]
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"monitoring_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump({"drift_score": drift_score, "quality": quality_metrics}, f, indent=2)

        # لاگ در MLflow
        with mlflow.start_run(run_name="monitoring") as monitor_run:
            mlflow.log_metric("drift_score", drift_score)
            for k, v in quality_metrics.items():
                mlflow.log_metric(f"quality_{k}", v)
            mlflow.log_artifact(str(report_path))

        logger.info("Monitoring report saved to %s", report_path)
        return {"drift_score": drift_score, "quality": quality_metrics}