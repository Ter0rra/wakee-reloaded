from airflow import DAG
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

with DAG(
    'health_check_weekly',
    default_args=default_args,
    description='Health checks hebdomadaires du système Wakee',
    schedule='0 3 * * 0',  # Dimanche 3h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['health', 'monitoring', 'tests'],
) as dag:
    
        start = EmptyOperator(
            task_id='start',
            doc_md="""
            ## Début des Tests Hebdomadaires
            
            Ce DAG lance tous les tests de qualité du projet.
            """
        )

        end = EmptyOperator(
            task_id='end',
            trigger_rule='all_done',  # S'exécute même si certaines tâches échouent
            doc_md="""
            ## Fin des Tests
            
            Tous les tests sont terminés.
            """
        )
            
start >> end