# ─────────────────────────────────────────────────────────────────────────────
# AC-CFM-GNN Dockerfile  (CPU)
#
# Für GPU-Support: docker build -f Dockerfile.gpu .
# ─────────────────────────────────────────────────────────────────────────────

FROM continuumio/miniconda3:24.1.2-0

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs build-essential && \
    git lfs install && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy env file first 
COPY configurations/local_env.yml ./configurations/local_env.yml

# Conda environment erstellen
RUN conda env create -f configurations/local_env.yml && \
    conda clean --all -f -y

# Conda env als Standard-Python setzen
ENV PATH="/opt/conda/envs/AC-CFM-GNN/bin:$PATH"
ENV CONDA_DEFAULT_ENV=AC-CFM-GNN

# Restlicher Code
COPY . .

# Ray Dashboard Port
EXPOSE 8887

CMD ["python", "src/main.py", "1", "1", "0", "8887"]
