FROM hailo-apps-core

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    zsh \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/tmp/* /var/lib/apt/lists/*

WORKDIR /tmp

COPY .devcontainer/requirements.txt .
RUN uv pip install -r requirements.txt

RUN rm -rf /tmp/*

WORKDIR /workspace

ENV SHELL=/usr/bin/zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
