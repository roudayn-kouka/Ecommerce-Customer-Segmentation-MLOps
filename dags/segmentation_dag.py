"""
Airflow DAG orchestrating the customer segmentation pipeline.

download -> unzip -> load -> 6 cleaning steps -> 7 feature steps
         -> scaling -> (PCA | correlation) -> model training

The project code is mounted at ``/opt/airflow/project`` (see docker-compose.yaml)
and every task calls the same functions as ``python -m src.pipeline``.
Each task returns the path of the file it produced; it is passed to the next
task through XCom.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src import pipeline
from src.clustering import train_and_save
from src.correlation import correlation_check
from src.data_loader import load_data
from src.download_data import ingest_data
from src.pca import pc_analyzer
from src.scaler import scaler
from src.unzip_data import unzip_file

DEFAULT_ARGS = {
    "owner": "roudayna",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _xcom(task_id: str) -> str:
    return "{{ ti.xcom_pull(task_ids='" + task_id + "') }}"


with DAG(
    dag_id="customer_segmentation",
    description="E-commerce customer segmentation: data pipeline + model training",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule=None,  # triggered manually (or by a sensor on new data)
    catchup=False,
    tags=["mlops", "segmentation"],
) as dag:
    download = PythonOperator(task_id="download_data", python_callable=ingest_data)
    unzip = PythonOperator(
        task_id="unzip_data",
        python_callable=unzip_file,
        op_kwargs={"zip_filename": _xcom("download_data")},
    )
    load = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
        op_kwargs={"excel_path": _xcom("unzip_data")},
    )
    download >> unzip >> load

    # Transaction cleaning: each step reads the previous step's output.
    previous = load
    for step in pipeline.CLEANING_STEPS:
        task = PythonOperator(
            task_id=step.__name__,
            python_callable=step,
            op_kwargs={"input_pickle_path": _xcom(previous.task_id)},
        )
        previous >> task
        previous = task
    clean_task_id = previous.task_id

    # Customer features: enrich the customer table using the clean transactions.
    rfm = PythonOperator(
        task_id="rfm",
        python_callable=pipeline.rfm,
        op_kwargs={"transactions_path": _xcom(clean_task_id)},
    )
    previous >> rfm
    previous = rfm
    for step in [
        pipeline.unique_products,
        pipeline.customers_behavior,
        pipeline.geographic_features,
        pipeline.cancellation_details,
        pipeline.seasonality_impacts,
    ]:
        task = PythonOperator(
            task_id=step.__name__,
            python_callable=step,
            op_kwargs={
                "customer_path": _xcom(previous.task_id),
                "transactions_path": _xcom(clean_task_id),
            },
        )
        previous >> task
        previous = task

    outliers = PythonOperator(
        task_id="removing_outlier",
        python_callable=pipeline.removing_outlier,
        op_kwargs={"input_pickle_path": _xcom(previous.task_id)},
    )
    scale = PythonOperator(
        task_id="scaler", python_callable=scaler, op_kwargs={"in_path": _xcom("removing_outlier")}
    )
    pca = PythonOperator(
        task_id="pca", python_callable=pc_analyzer, op_kwargs={"in_path": _xcom("scaler")}
    )
    corr = PythonOperator(
        task_id="correlation",
        python_callable=correlation_check,
        op_kwargs={"in_path": _xcom("scaler")},
    )
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_and_save,
        op_kwargs={"in_path": _xcom("removing_outlier")},
    )

    previous >> outliers >> scale >> [pca, corr]
    outliers >> train
