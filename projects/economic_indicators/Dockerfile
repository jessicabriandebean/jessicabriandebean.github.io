FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY envs/economic_indicators/pyproject.toml .
COPY envs/economic_indicators/uv.lock .

RUN uv sync --frozen

COPY projects/economic_indicators .

EXPOSE 8501

CMD ["/app/.venv/bin/streamlit", "run", "app/econ_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]