FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY ../../envs/portfolio_optimization/pyproject.toml .
COPY ../../envs/portfolio_optimization/uv.lock .

RUN uv sync --frozen

COPY . .

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]