# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

FROM ${PYTHON_IMAGE} AS model
ARG MODEL_REPOSITORY=MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c
ARG MODEL_REVISION=b2730f16019076bb0009481121efbe4705e0e378
COPY requirements-model.lock model.sha256 /tmp/
RUN pip install --no-cache-dir --only-binary :all: --require-hashes -r /tmp/requirements-model.lock
RUN python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir='/opt/model', allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'spm.model'])" "$MODEL_REPOSITORY" "$MODEL_REVISION" \
 && rm -rf /opt/model/.cache \
 && cd /opt/model && sha256sum -c /tmp/model.sha256 \
 && test "$(ls | wc -l)" -eq "$(wc -l < /tmp/model.sha256)"

FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf \
    NLI_MODEL_DIR=/opt/model NLI_HOST=0.0.0.0 NLI_PORT=8080
COPY requirements.lock requirements-torch.lock /tmp/
RUN pip install --no-cache-dir --only-binary :all: --require-hashes --no-deps -r /tmp/requirements.lock \
 && pip install --no-cache-dir --only-binary :all: --require-hashes --no-deps --index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-torch.lock \
 && pip check \
 && rm /tmp/requirements.lock /tmp/requirements-torch.lock
COPY --from=model /opt/model /opt/model
COPY nli_server /app/nli_server
WORKDIR /app
USER 65532:65532
EXPOSE 8080
CMD ["python", "-m", "nli_server.server"]
