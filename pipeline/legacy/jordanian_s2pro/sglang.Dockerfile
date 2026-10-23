FROM lmsysorg/sglang:v0.5.16
# This image is independent of the working Fish conda environment.
# Copy the installer without running pip against Debian's managed interpreter.
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv
RUN uv venv --python python3 --system-site-packages /opt/sglang-venv
ENV VIRTUAL_ENV=/opt/sglang-venv
ENV PATH="/opt/sglang-venv/bin:${PATH}"
COPY sglang-overrides.txt /opt/sglang-overrides.txt
RUN uv pip install --python /opt/sglang-venv/bin/python --prerelease=allow \
      --override /opt/sglang-overrides.txt \
      --reinstall-package sglang-omni \
      'sglang-omni==0.1.3' 'descript-audio-codec==1.0.0' \
      'descript-audiotools==0.7.2' 'protobuf>=6.31.1,<7.0.0'
COPY sglang_build_check.py /opt/sglang_build_check.py
RUN python3 /opt/sglang_build_check.py
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT []

