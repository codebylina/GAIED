# Third-party code

`falda_distiller.py` is copied **unchanged** from FALDA
(https://github.com/rick-stevens-ai/falda, commit `ff6d83b`), Apache License 2.0
(see `LICENSE.falda`). It is FALDA's own T0 -> T1 -> T2 -> T3 distillation sidecar; this
baseline runs it as a separate process, configured only through its environment variables.

The FALDA gateway itself (TypeScript) is not copied: `docker/falda/Dockerfile` builds it
from the same commit of the official repository.
