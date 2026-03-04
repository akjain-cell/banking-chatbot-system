FROM python:3.11-slim

WORKDIR /app

#installsystem dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#Copyrequirements and installPythondependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copyapplication code
COPY . .

#Download theembeddingmodelduring build (cached)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

#Expose port
EXPOSE 7860

#Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
