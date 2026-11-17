"""
Run the analysis service (independently of the scientific agents).

  python -m analysis run      [--namespace NS] [--seconds N]     online: memory events + Flowcept MQ / DB
  python -m analysis offline  [--namespace NS] --buffers 'output/flowcept_buffer_*.jsonl'   one pass, post-campaign
  python -m analysis findings [--namespace NS]                    print the findings (M2) and guidance (M4)

Environment: REDIS_HOST, MEMORY_NAMESPACE (default "gaied"), ANALYSIS_CONSUMER (default "analysis-1"),
ANALYSIS_LLM=true for LLM explanations; Flowcept settings via FLOWCEPT_SETTINGS_PATH (online settings
enable the MQ and the MongoDB catch-up).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from memory import SharedMemory, Store
from memory.provenance import FlowceptSource

from .detectors import AnalysisConfig
from .diagnosis import explainer_from_env
from .service import AnalysisService


def build(namespace: str, domain_name: str) -> AnalysisService:
    if domain_name == "chemistry":
        from memory.domains.chemistry import ChemistryDomain
        domain = ChemistryDomain()
    else:
        from memory.domain import Domain
        domain = Domain()
    store = Store.connect(namespace, host=os.environ.get("REDIS_HOST", "localhost"))
    memory = SharedMemory(store, domain, agent="analysis")
    return AnalysisService(memory, AnalysisConfig(), FlowceptSource(domain),
                           consumer=os.environ.get("ANALYSIS_CONSUMER", "analysis-1"),
                           period_s=float(os.environ.get("ANALYSIS_PERIOD_S", "15")),
                           explainer=explainer_from_env())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m analysis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "offline", "findings"])
    ap.add_argument("--namespace", default=os.environ.get("MEMORY_NAMESPACE", "gaied"))
    ap.add_argument("--domain", default="chemistry")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--buffers", default="output/flowcept_buffer_*.jsonl")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    svc = build(a.namespace, a.domain)
    if a.command == "run":
        svc.run(max_seconds=a.seconds)
    elif a.command == "offline":
        svc.recover()
        while svc.read_events(block_ms=10):
            pass
        n = svc.ingest_buffers(a.buffers)
        n += svc.catchup()
        print(f"{n} Flowcept tasks ingested")
        for f in svc.analyze():
            print(f)
        print(svc.feedback.evaluate_guidance())
    m = svc.m
    for e in m.store.find("M2"):
        if e.created_by == "analysis":
            print(f"[M2 {e.kind}/{e.status} x{e.occurrences}] {e.summary}"
                  + (f"  CAUSE: {e.confirmed_cause}" if e.confirmed_cause else ""))
    for g in m.store.find("M4"):
        print(f"[M4 {g.kind}/{g.status} -> {g.targets}] {g.text}  delivery={m.guidance.delivery(g.id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
