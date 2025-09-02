FROM python:3.12.1-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY api/ /app/api
COPY api/requirements.txt .
COPY data/artifacts/preprocessor.pkl /app/data/artifacts/preprocessor.pkl

RUN pip install --upgrade pip && pip install -r requirements.txt 

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]