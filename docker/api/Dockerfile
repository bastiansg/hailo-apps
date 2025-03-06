FROM hailo-apps-core

COPY hailo_apps /src/hailo_apps
COPY requirements.txt pyproject.toml  /src/
RUN --mount=type=bind,source=.git,target=/src/.git \
    uv pip install /src --no-deps --system

ARG API_PORT
WORKDIR /src/hailo_apps/api
CMD fastapi run app.py --port ${API_PORT}
