FROM python:3.12
WORKDIR /usr/local/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
CMD ["fastapi", "run", "server.py", "--port", "5042"]
