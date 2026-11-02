FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates git ripgrep fd-find jq file unzip zip fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.14
COPY python/ /opt/python/
RUN cd /opt/python && uv sync --frozen --no-dev --no-install-project --no-python-downloads \
    && ln -s /opt/python/.venv/lib/python3.12/site-packages/nodejs_wheel/bin/node /usr/local/bin/node
ENV PATH="/opt/python/.venv/bin:/usr/local/bin:/usr/bin:/bin"

WORKDIR /opt/pi
COPY manifests/ ./
RUN node /opt/python/.venv/lib/python3.12/site-packages/nodejs_wheel/lib/node_modules/npm/bin/npm-cli.js \
    ci --ignore-scripts --include=dev --no-audit --no-fund
COPY source/ ./
COPY models.all.json /opt/models.all.json
RUN node packages/ai/scripts/hydrate-model-catalog.ts /opt/models.all.json \
    && rm /opt/models.all.json
COPY entrypoint.mjs /opt/pi-entrypoint.mjs
ENV HOME=/home/pi PI_CODING_AGENT_DIR=/home/pi/.pi/agent OPENAI_API_KEY=pi-sandbox-relay
WORKDIR /workspace
USER 65534:65534
ENTRYPOINT ["node", "/opt/pi-entrypoint.mjs"]
