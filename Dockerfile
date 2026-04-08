FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_HOME=/opt/hf-home
ENV TRANSFORMERS_CACHE=/opt/hf-home
ENV ENABLE_WEB_INTERFACE=true
RUN mkdir -p /opt/hf-home && python - <<'PY'
from server.semantics import DEFAULT_ANALYZER
print(f"Semantic backend ready: {DEFAULT_ANALYZER._backend}")
PY
ENV PORT=7860
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
