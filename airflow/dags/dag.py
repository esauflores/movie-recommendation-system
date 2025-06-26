import sys
sys.path.append("/opt/airflow/data")

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from load_data import download_dataset, move_dataset_contents
from preprocess_data import main

# from job2 import run as run_job2

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_job1():
    path = download_dataset(force_download=True)
    data_dir = "data/raw"
    move_dataset_contents(path, data_dir)

def run_job2():
    main()

with DAG(
    dag_id='weekly_movie_jobs',
    default_args=default_args,
    description='Run weekly ML/data processing jobs',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'weekly'],
) as dag:

    job1_task = PythonOperator(
        task_id='run_job1',
        python_callable=run_job1,
    )

    job2_task = PythonOperator(
        task_id='run_job2',
        python_callable=run_job2,
    )

    job1_task >> job2_task
