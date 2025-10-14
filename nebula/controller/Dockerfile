FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

# Set timezone
RUN apt-get update && apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/Madrid /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Install Python 3.11 directly from Ubuntu repos (no PPA needed)
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils python3.11-venv \
    software-properties-common curl net-tools iproute2 iputils-ping

# Update alternatives to make Python 3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install build tools, git, and compilers
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ clang git make cmake g++-aarch64-linux-gnu dos2unix

# Install Docker CLI + plugins inside container
RUN apt-get install -y ca-certificates curl gnupg \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
 && chmod a+r /etc/apt/keyrings/docker.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | tee /etc/apt/sources.list.d/docker.list > /dev/null \
 && apt-get update \
 && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Install uv (Python package manager)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Copy project dependencies
COPY pyproject.toml .

# Install Python 3.11.7 via uv and pin it
RUN uv python install 3.11.7 \
 && uv python pin 3.11.7 \
 && uv sync --group controller

ENV PATH="/.venv/bin:$PATH"

# Copy startup script
COPY /nebula/controller/start_services.sh .
RUN dos2unix start_services.sh && chmod +x start_services.sh

ENTRYPOINT ["/bin/bash", "/start_services.sh"]
