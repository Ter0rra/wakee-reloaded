from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from datetime import timedelta, datetime

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def say_hi(**context):
        """need to try to say hello"""
        print('hello word !')

with DAG(
    'dag_retrain',
    default_args=default_args,
    description='fake_retrain',
    schedule='0 3 * * 0',  # Dimanche 3h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['retrain', 'fake', 'tests'],
) as dag:
    
        start = EmptyOperator(
            task_id='start',
            doc_md="""
            ## Début des Tests Hebdomadaires
            
            Ce DAG lance tous les tests de qualité du projet.
            """
        )

        try_it_out = PythonOperator(
                task_id='try',
                python_callable=say_hi
        )

        end = EmptyOperator(
            task_id='end',
            trigger_rule='all_done',  # S'exécute même si certaines tâches échouent
            doc_md="""
            ## Fin des Tests
            
            Tous les tests sont terminés.
            """
        )
            
start >> try_it_out >> end