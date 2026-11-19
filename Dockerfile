FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser ai/ ./ai/
COPY --chown=appuser:appuser game/ ./game/
COPY --chown=appuser:appuser players/ ./players/
COPY --chown=appuser:appuser ui/ ./ui/
COPY --chown=appuser:appuser main.py ./

USER appuser

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/', timeout=2)"]

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
