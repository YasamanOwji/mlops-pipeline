"""
workflow.py
-----------
Defines a scheduled workflow using Prefect.
"""

from prefect import flow
from prefect.schedules import CronSchedule
from run_pipeline import mlops_pipeline
from monitor import main as monitor_main
import logging

logger = logging.getLogger(__name__)


@flow(name="scheduled_mlops", schedule=CronSchedule(cron="0 0 * * 1"))  # هر دوشنبه ساعت 00:00
def scheduled_mlops():
    """Executes the full pipeline and then runs monitoring."""
    run_id = mlops_pipeline(environment="production")
    logger.info("Pipeline executed with run_id=%s", run_id)
    monitor_main()
    logger.info("Monitoring completed.")


if __name__ == "__main__":
    scheduled_mlops()