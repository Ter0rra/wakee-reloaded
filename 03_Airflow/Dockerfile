FROM apache/airflow:3.1.7
ADD requirements.txt .

RUN pip install --upgrade pip
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt