FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for SUMO
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libxerces-c-dev \
    libproj-dev \
    libgdal-dev \
    libfox-1.6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SUMO from package manager
RUN apt-get update && apt-get install -y sumo sumo-tools && \
    rm -rf /var/lib/apt/lists/*

# Add SUMO to PATH
ENV PATH="/usr/share/sumo/bin:${PATH}"
ENV SUMO_HOME="/usr/share/sumo"

# PyTorch: CPU index only (smaller/faster than default CUDA wheels; avoids pip timeouts on slow links)
ENV PIP_DEFAULT_TIMEOUT=1200
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --retries 10 -r requirements.txt \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
