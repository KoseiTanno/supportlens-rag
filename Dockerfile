FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home app
COPY --chown=app:app pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev
COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app data ./data
USER app
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src
CMD ["uvicorn", "supportlens.api:app", "--host", "0.0.0.0", "--port", "8000"]
