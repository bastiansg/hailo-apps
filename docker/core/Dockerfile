ARG UV_VERSION=0.9.24
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_source

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gpg \
    && curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.org/debian bookworm main" > /etc/apt/sources.list.d/raspi.list \
    && apt-get update

ARG PYTHON_VERSION=3.11
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    git-core \
    openssh-client \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-setuptools \
    python3-picamera2 \
    python3-rpi.gpio \
    hailo-all \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/tmp/* /var/lib/apt/lists/*

RUN rm /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

ENV UV_SYSTEM_PYTHON=1
COPY --from=uv_source /uv /uvx /bin/

WORKDIR /tmp

COPY uv.toml .
COPY ./requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache \
    uv pip install -r requirements.txt

RUN rm -rf /tmp/*
WORKDIR /root

ENV HAILORT_LOGGER_PATH=NONE
