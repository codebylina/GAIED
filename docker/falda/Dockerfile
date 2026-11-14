# FALDA gateway (https://github.com/rick-stevens-ai/falda), built from a pinned commit.
# Used as the shared memory of the GAIED agents in the FALDA baseline.
FROM node:22-bookworm-slim

ARG FALDA_COMMIT=ff6d83b
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates python3 make g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN git clone https://github.com/rick-stevens-ai/falda.git && cd falda && git checkout ${FALDA_COMMIT}
WORKDIR /opt/falda
RUN npm ci --no-audit --no-fund && npm run build

ENV FALDA_PORT=8077 FALDA_ROOT=/data FALDA_DIM=768
EXPOSE 8077
CMD ["node", "dist/gateway.js"]
