# ==============================================================================
# STAGE 1 : Builder
# ==============================================================================
FROM continuumio/miniconda3:23.10.0-1 AS builder

WORKDIR /build

# Copie les fichiers de dépendances
COPY environment.yml .

# Créer l'environnement
RUN conda env create -f environment.yml && \
    conda clean -afy

# ==============================================================================
# STAGE 2 : Runtime (image finale légère)
# ==============================================================================
FROM continuumio/miniconda3:23.10.0-1

WORKDIR /app

# Copier l'environnement depuis le builder
COPY --from=builder /opt/conda/envs/wakee_api /opt/conda/envs/wakee_api

# Copier l'application
COPY app.py .

# Expose port
EXPOSE 7860

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV ORT_DISABLE_TELEMETRY=1
ENV PATH=/opt/conda/envs/wakee_api/bin:$PATH

# Activer l'environnement et démarrer
SHELL ["conda", "run", "-n", "wakee_api", "/bin/bash", "-c"]

CMD ["conda", "run", "--no-capture-output", "-n", "wakee_api", \
     "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]