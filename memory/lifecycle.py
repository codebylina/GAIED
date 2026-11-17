"""
Memory lifecycle: expiration, staleness, retention / archival, and quality metrics.

Nothing is deleted: expired or superseded items are retired or archived, with the reason in their
version history. Run periodically (the analysis service does it; `python -m memory maintain` too).
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

from .store import Store


@dataclass
class RetentionPolicy:
    proposed_guidance_ttl_s: float = 3600.0      # a proposed M4 item nobody activated is retired
    stale_after_new_results: int = 10            # M3 re-evaluated when its condition got this many new results
    unverifiable_max_age_s: float = 86400.0      # unchecked M3 hypotheses become stale after this
    archive_refuted_after_s: float = 7 * 86400.0
    archive_retired_after_s: float = 7 * 86400.0
    archive_resolved_after_s: float = 7 * 86400.0


class Lifecycle:
    def __init__(self, memory, policy: RetentionPolicy | None = None) -> None:
        self.m, self.policy = memory, policy or RetentionPolicy()

    @property
    def s(self) -> Store:
        return self.m.store

    def run(self, by: str = "lifecycle") -> dict:
        now, p, report = time.time(), self.policy, Counter()
        # M4: expiration of guidance
        for g in self.s.find("M4", status=["active", "validated", "proposed"]):
            if g.expires_at and g.expires_at < now:
                self.m.guidance.set_status(g.id, "retired", by, "expired")
                report["M4.expired"] += 1
            elif g.status == "proposed" and now - g.created_at > p.proposed_guidance_ttl_s:
                self.m.guidance.set_status(g.id, "retired", by, "proposed and never activated")
                report["M4.retired_unactivated"] += 1
        # M3: staleness and re-evaluation
        for k in self.m.knowledge.active():
            checkable = bool(k.claim and k.claim.get("type") not in (None, "none"))
            if checkable:
                done_at = k.evaluation.get("n_results_at_eval", 0)
                if self.m.results.count(k.condition) - done_at >= p.stale_after_new_results:
                    self.m.knowledge.reevaluate(k.id, by=by)
                    report["M3.reevaluated_stale"] += 1
            elif k.status == "proposed" and now - k.created_at > p.unverifiable_max_age_s:
                self.s.update("M3", k.id, lambda r: setattr(r, "status", "stale"), by,
                              "unverifiable and unconfirmed for too long")
                report["M3.stale"] += 1
        # M2: expiring episodes
        for e in self.s.find("M2"):
            if e.expires_at and e.expires_at < now:
                self.s.archive("M2", e.id, by, "expired")
                report["M2.archived_expired"] += 1
            elif e.status in ("resolved", "dismissed") and now - e.updated_at > p.archive_resolved_after_s:
                self.s.archive("M2", e.id, by, "resolved long ago")
                report["M2.archived_resolved"] += 1
        # archival of refuted knowledge and retired guidance
        for k in self.s.find("M3", status="refuted"):
            if now - k.updated_at > p.archive_refuted_after_s:
                self.s.archive("M3", k.id, by, "refuted long ago")
                report["M3.archived"] += 1
        for g in self.s.find("M4", status=["retired", "ineffective"]):
            if now - g.updated_at > p.archive_retired_after_s:
                self.s.archive("M4", g.id, by, "retired long ago")
                report["M4.archived"] += 1
        return dict(report)

    def quality(self) -> dict:
        """Status distribution per layer, contradiction rate, guidance effectiveness, usage."""
        out = {}
        for L in ("M1", "M2", "M3", "M4"):
            recs = self.s.get_many(L, self.s.ids(L))
            out[L] = {"active": len(recs), "by_status": dict(Counter(r.status for r in recs)),
                      "by_kind": dict(Counter(r.kind for r in recs)), "archived": len(self.s.ids(L, archived=True))}
        m3 = out["M3"]["by_status"]
        judged = sum(m3.get(s, 0) for s in ("supported", "validated", "contested", "refuted"))
        out["M3"]["contradiction_rate"] = round((m3.get("contested", 0) + m3.get("refuted", 0)) / judged, 3) if judged else None
        m4 = out["M4"]["by_status"]
        rated = m4.get("validated", 0) + m4.get("ineffective", 0)
        out["M4"]["effective_rate"] = round(m4.get("validated", 0) / rated, 3) if rated else None
        out["metrics"] = self.s.metrics()
        return out
