"""
run_pipeline.py
-----------------
Runs the full MLOps pipeline end-to-end using Prefect for orchestration.
"""

from prefect import flow, task
from pipeline import MLOpsPipeline
from generate_data import generate_dataset
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
DATA_PATH = root_dir / "data" / "gamified_learning_engagement.csv"


@task
def generate_and_save_data():
    data_dir = DATA_PATH.parent
    data_dir.mkdir(exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    return str(DATA_PATH)


@task
def version_data(pipeline, data_path):
    return pipeline.version_data(data_path)


@task
def train_and_register(pipeline, data_path, data_version):
    model_params = pipeline.config["model"]["params"]
    return pipeline.train_and_register(data_path, data_version, model_params)


@task
def deploy_model(pipeline, run_id, environment="staging"):
    return pipeline.deploy_model(run_id, environment)


@flow(name="mlops_pipeline")
def mlops_pipeline(environment="staging"):
    pipeline = MLOpsPipeline()

    data_path = generate_and_save_data()
    data_version = version_data(pipeline, data_path)
    run_id = train_and_register(pipeline, data_path, data_version)
    deploy_path = deploy_model(pipeline, run_id, environment)

    print(f"✅ Pipeline complete. run_id={run_id}, deployed to {deploy_path}")
    return run_id


if __name__ == "__main__":
    mlops_pipeline(environment="staging")