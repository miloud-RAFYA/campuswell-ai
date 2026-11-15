
FROM apache/airflow:3.4.0-python3.14

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir \
    "apache-airflow==3.4.0" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.14.txt" \
    -r /requirements.txt
