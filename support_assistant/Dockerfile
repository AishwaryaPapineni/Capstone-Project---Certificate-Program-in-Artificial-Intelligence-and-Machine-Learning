FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# MOCK_LLM defaults to "1" (the graded baseline) unless overridden at `docker run`
# with -e MOCK_LLM=0 (and -e GROQ_API_KEY=... for the optional real-LLM extension).
ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
