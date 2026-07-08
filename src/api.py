"""
monitor.py
----------
Runs monitoring using the pipeline class.
"""

from pipeline import MLOpsPipeline
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    pipeline = MLOpsPipeline()
    production_model_path = pipeline.root_dir / pipeline.config["deployment"]["production_dir"]

    # اگر داده‌ی جدید وجود نداشت، از داده‌ی تست استفاده کن
    new_data_path = pipeline.root_dir / "data" / "new_data.csv"
    if not new_data_path.exists():
        logger.warning("No new data found. Using test data as proxy.")
        new_data_path = pipeline.root_dir / "data" / "test_data.csv"
        if not new_data_path.exists():
            logger.error("No test data found either. Skipping monitoring.")
            return

    result = pipeline.run_monitoring(str(production_model_path), str(new_data_path))
    logger.info("Monitoring results: %s", result)

if __name__ == "__main__":
    main()