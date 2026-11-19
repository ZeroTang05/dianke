ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_ROOT=/input \
    OUTPUT_DIR=/output

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/program/solution
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY run.sh ./
COPY src/ ./src/
RUN chmod +x run.sh

ENTRYPOINT ["/app/program/solution/run.sh"]
