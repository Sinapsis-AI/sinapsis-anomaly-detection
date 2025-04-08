FROM sinapsis-nvidia:base

COPY . /app/
WORKDIR /app
RUN uv sync --frozen

RUN uv pip install sinapsis-anomaly-detection[all] --extra-index-url https://pypi.sinapsis.tech