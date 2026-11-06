# Image pour faire tourner GAIED (Academy + LangGraph + Flowcept + xTB) sur CPU.
# xTB et RDKit viennent de conda-forge ; le reste de pip.
FROM mambaorg/micromamba:1.5.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 + RDKit + xTB (conda-forge)
RUN micromamba install -y -n base -c conda-forge python=3.12 pip rdkit xtb-python \
    && micromamba clean -a -y

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Dépendances Python de GAIED, en deux étapes séparées : si la connexion coupe
# pendant la 2e, Docker garde la 1re et ne retente que Flowcept.
RUN pip install --no-cache-dir --retries 10 --timeout 60 \
    "academy-py>=0.3.0" parsl ase "langchain>=1.0.0" langchain-openai langgraph \
    requests redis msgpack pymongo pyarrow

# Flowcept depuis le fork d'Amal (le pyproject de GAIED pointe vers un lien anonymisé).
RUN pip install --no-cache-dir --retries 10 --timeout 60 \
    "flowcept[redis,mongo] @ git+https://github.com/GueroudjiAmal/flowcept.git"

# GAIED a été écrit pour Academy 0.3/0.4 : la version 1.0 a retiré init_logging.
# La 0.4.0 garde aussi les méthodes de Runtime que le plugin Flowcept modifie.
RUN pip install --no-cache-dir --retries 10 --timeout 60 "academy-py==0.4.0"

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OMP_STACKSIZE=1G

WORKDIR /app
CMD ["python", "run-07.py"]
