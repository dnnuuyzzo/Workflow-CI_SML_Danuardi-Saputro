FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY model/ /app/model/

# Copy serving script
COPY serve.py .

EXPOSE 5001

CMD ["python", "serve.py"]
