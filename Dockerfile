FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/temp

EXPOSE 5002

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5002"]