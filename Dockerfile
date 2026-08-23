FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv pip install --no-deps -e .

ENV PATH="/app/.venv/bin:$PATH"

CMD ["tkf", "controller"]
