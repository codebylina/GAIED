# Image to run GAIED (Academy + LangGraph + Flowcept + xTB) on CPU.
# xTB and RDKit come from conda-forge; everything else from pip.
FROM mambaorg/micromamba:1.5.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 + RDKit + xTB (conda-forge)
RUN micromamba install -y -n base -c conda-forge python=3.12 pip rdkit xtb-python \
    && micromamba clean -a -y

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# GAIED Python dependencies, in two separate layers: if the connection drops
# during the second one, Docker keeps the first and only retries Flowcept.
RUN pip install --no-cache-dir --retries 10 --timeout 60 \
    "academy-py>=0.3.0" parsl ase "langchain>=1.0.0" langchain-openai langgraph \
    requests redis msgpack pymongo pyarrow

# Flowcept from Amal's fork (GAIED's pyproject points to an anonymized link).
RUN pip install --no-cache-dir --retries 10 --timeout 60 \
    "flowcept[redis,mongo] @ git+https://github.com/GueroudjiAmal/flowcept.git"

# GAIED was written for Academy 0.3/0.4: version 1.0 removed init_logging.
# 0.4.0 also keeps the Runtime methods that the Flowcept plugin patches.
RUN pip install --no-cache-dir --retries 10 --timeout 60 "academy-py==0.4.0"

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OMP_STACKSIZE=1G

WORKDIR /app
CMD ["python", "run-07.py"]
