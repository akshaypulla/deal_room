FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_HOME=/opt/hf-home
ENV TRANSFORMERS_CACHE=/opt/hf-home
ENV ENABLE_WEB_INTERFACE=true
RUN mkdir -p /opt/hf-home && python - <<'PY'
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
except Exception as exc:
    print(f"Model pre-cache skipped: {exc}")
PY
ENV PORT=7860
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
