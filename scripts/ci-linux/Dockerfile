# Local stand-in for the GitHub CI Linux jobs: x86_64 Debian with Python ${PY}, FFmpeg,
# the Qt runtime libraries, and this project's dependencies. Built by scripts/ci-linux/run.sh.
ARG PY=3.12
FROM --platform=linux/amd64 python:${PY}-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg libegl1 libxkbcommon0 libfontconfig1 libdbus-1-3 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Dependencies come from the same pyproject.toml as CI. The project itself is installed
# at run time from a copy of the working tree.
COPY pyproject.toml README.md LICENSE /deps/
COPY src/audio_from_video/__init__.py /deps/src/audio_from_video/__init__.py
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip \
 && pip install --no-cache-dir --root-user-action=ignore "/deps[dev,gui,test-gui]" \
 && pip uninstall -y --root-user-action=ignore audio-from-video

ENV QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 \
    PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /work
