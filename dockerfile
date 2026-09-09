FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN uv pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]