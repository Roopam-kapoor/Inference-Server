FROM vllm/vllm-openai:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .

EXPOSE 8001

ENTRYPOINT []

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8001"]