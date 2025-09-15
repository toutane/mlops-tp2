docker run \
    --name mlflow \
    -it\
    --rm \
    -p 8000:8000 \
    --network mlops-tp2 \
    ghcr.io/mlflow/mlflow:v3.4.0rc0 \
    mlflow server \
    --host 0.0.0.0 \
    --port 8000

