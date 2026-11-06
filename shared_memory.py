"""
shared_memory.py — Mémoire partagée entre agents GAIED, stockée dans Redis.

Idée (schéma d'Amal) : au lieu que chaque agent garde ses résultats dans sa RAM
(`molecule_cache`, perdu au crash et invisible pour les autres), les agents POUSSENT
ce qu'ils font et apprennent dans une mémoire commune, et la RELISENT avant de décider.

Ce que contient la mémoire d'une campagne (toutes les clés commencent par mem:{campagne}:) :

  goal                  JSON     objectif de la campagne (écrit par run-07.py ou un humain)
  claim:{mol}           chaîne   VERROU « je simule cette molécule » (avec expiration)
  result:{mol}          hash     résultat : energy, smiles, agent, task_id (Flowcept), status, time
  ranking               zset     molécules réussies, classées par énergie
  failed                set      molécules dont la simulation a échoué (SMILES invalide…)
  lessons               stream   conclusions et leçons écrites par les agents
  instructions          stream   consignes des agents d'analyse (travail de Nadjiba)
  stats                 hash     compteurs : simulations faites, résultats réutilisés, attentes…
  events                canal    notification pub/sub à chaque écriture

{mol} est la forme canonique du SMILES (calculée avec RDKit si disponible), pour que deux
écritures différentes de la même molécule soient reconnues comme identiques.

Les trois opérations de base :
  PUSH  : push_result, push_lesson, push_instruction, set_goal
  PULL  : get_result, pull_results, best, pull_lessons, pull_instructions, get_goal,
          render_for_prompt (résumé borné à mettre dans le prompt du LLM)
  COORDINATION : claim / release, et get_or_compute qui combine tout

Outil en ligne de commande (pour voir la mémoire) :
  python shared_memory.py campaigns          liste les campagnes en mémoire
  python shared_memory.py show [campagne]    résumé d'une campagne (la plus récente par défaut)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

try:  # RDKit est présent dans l'image Docker ; sans lui on garde le SMILES tel quel
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = None


def canonical_smiles(smiles: str) -> str:
    """Forme canonique d'un SMILES (ou le SMILES nettoyé s'il est invalide / sans RDKit)."""
    smiles = smiles.strip()
    if Chem is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    return smiles


class SimulationFailed(Exception):
    """La simulation de cette molécule a déjà échoué (connu grâce à la mémoire)."""


class SharedMemory:
    def __init__(self, client, campaign_id: str = "default", agent_id: str = "agent",
                 claim_ttl: int = 600, prefix: str = "mem") -> None:
        self.r = client
        self.campaign_id = campaign_id
        self.agent_id = agent_id
        self.claim_ttl = claim_ttl
        self.p = f"{prefix}:{campaign_id}"

    # ------------------------------------------------------------------ création
    @classmethod
    def from_env(cls, agent_id: str = "agent") -> "SharedMemory | None":
        """Crée la mémoire à partir des variables d'environnement, ou renvoie None si elle
        est désactivée (SHARED_MEMORY=false) ou si Redis est injoignable."""
        if os.environ.get("SHARED_MEMORY", "true").lower() not in ("1", "true", "yes"):
            logger.info("Mémoire partagée désactivée (SHARED_MEMORY=false).")
            return None
        try:
            import redis
            client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"),
                                 port=int(os.environ.get("REDIS_PORT", "6379")),
                                 decode_responses=True)
            client.ping()
        except Exception as exc:  # Redis absent : l'agent continue sans mémoire partagée
            logger.warning("Mémoire partagée indisponible (%s) : l'agent continue sans.", exc)
            return None
        campaign = (os.environ.get("MEMORY_CAMPAIGN_ID")
                    or os.environ.get("FLOWCEPT_CAMPAIGN_ID") or "default")
        ttl = int(os.environ.get("MEMORY_CLAIM_TTL", "600"))
        mem = cls(client, campaign_id=campaign, agent_id=agent_id, claim_ttl=ttl)
        client.zadd("mem:campaigns", {campaign: time.time()})   # index des campagnes
        return mem

    def k(self, *parts: str) -> str:
        return ":".join((self.p, *parts))

    def _notify(self, event: str, **data: Any) -> None:
        self.r.publish(self.k("events"), json.dumps({"event": event, "agent": self.agent_id, **data}))

    def _count(self, field: str, n: int = 1) -> None:
        self.r.hincrby(self.k("stats"), field, n)

    # ------------------------------------------------------------------ objectif
    def set_goal(self, goal: dict) -> None:
        self.r.set(self.k("goal"), json.dumps(goal))
        self._notify("goal")

    def get_goal(self) -> dict | None:
        raw = self.r.get(self.k("goal"))
        return json.loads(raw) if raw else None

    # ------------------------------------------------------------------ verrous
    def claim(self, smiles: str, ttl: int | None = None) -> bool:
        """Prend le verrou sur une molécule. True si on l'a obtenu, False si un autre l'a."""
        key = self.k("claim", canonical_smiles(smiles))
        return bool(self.r.set(key, self.agent_id, nx=True, ex=ttl or self.claim_ttl))

    def claimed_by(self, smiles: str) -> str | None:
        return self.r.get(self.k("claim", canonical_smiles(smiles)))

    def release(self, smiles: str) -> None:
        """Libère le verrou, seulement s'il nous appartient."""
        key = self.k("claim", canonical_smiles(smiles))
        if self.r.get(key) == self.agent_id:
            self.r.delete(key)

    # ------------------------------------------------------------------ résultats
    def push_result(self, smiles: str, energy: float | None, task_id: str | None = None,
                    error: str | None = None) -> None:
        mol = canonical_smiles(smiles)
        ok = error is None and energy is not None and math.isfinite(energy)
        record = {
            "smiles": smiles, "canonical": mol, "agent": self.agent_id,
            "task_id": task_id or "", "time": time.time(),
            "status": "ok" if ok else "error",
            "energy": energy if ok else "",
            "error": "" if ok else (error or f"energy={energy}"),
        }
        self.r.hset(self.k("result", mol), mapping=record)
        if ok:
            self.r.zadd(self.k("ranking"), {mol: energy})
        else:
            self.r.sadd(self.k("failed"), mol)
        self._notify("result", smiles=mol, status=record["status"])

    def get_result(self, smiles: str) -> dict | None:
        rec = self.r.hgetall(self.k("result", canonical_smiles(smiles)))
        if not rec:
            return None
        if rec.get("status") == "ok":
            rec["energy"] = float(rec["energy"])
        return rec

    def pull_results(self) -> dict[str, float]:
        """Toutes les molécules simulées avec succès, par n'importe quel agent : {smiles: énergie}."""
        return {m: e for m, e in self.r.zrange(self.k("ranking"), 0, -1, withscores=True)}

    def failed_molecules(self) -> set[str]:
        return set(self.r.smembers(self.k("failed")))

    def best(self, n: int = 5) -> list[tuple[str, float]]:
        return self.r.zrevrange(self.k("ranking"), 0, n - 1, withscores=True)

    # ------------------------------------------------------------------ leçons
    def push_lesson(self, text: str, kind: str = "conclusion", evidence: list[str] | None = None) -> None:
        self.r.xadd(self.k("lessons"), {
            "agent": self.agent_id, "kind": kind, "text": text,
            "evidence": json.dumps(evidence or []), "time": time.time(),
        }, maxlen=2000, approximate=True)
        self._notify("lesson", kind=kind)

    def pull_lessons(self, count: int = 10, exclude_own: bool = False) -> list[dict]:
        """Les leçons les plus récentes (de la plus ancienne à la plus récente)."""
        entries = self.r.xrevrange(self.k("lessons"), count=count * 3 if exclude_own else count)
        out = [dict(fields, id=eid) for eid, fields in entries
               if not (exclude_own and fields.get("agent") == self.agent_id)]
        return list(reversed(out[:count]))

    # ------------------------------------------------------------------ instructions
    def push_instruction(self, text: str, target: str = "all", status: str = "info",
                         evidence: list[str] | None = None, author: str = "analysis") -> None:
        """Écrite par les agents d'analyse (Nadjiba). target = un agent_id, ou "all"."""
        self.r.xadd(self.k("instructions"), {
            "author": author, "target": target, "status": status, "text": text,
            "evidence": json.dumps(evidence or []), "time": time.time(),
        }, maxlen=2000, approximate=True)
        self._notify("instruction", target=target)

    def pull_instructions(self, count: int = 5) -> list[dict]:
        """Instructions récentes qui concernent cet agent (ou tous les agents)."""
        entries = self.r.xrevrange(self.k("instructions"), count=count * 5)
        mine = [dict(f, id=eid) for eid, f in entries if f.get("target") in ("all", self.agent_id)]
        return list(reversed(mine[:count]))

    # ------------------------------------------------------------------ résumé pour le LLM
    def render_for_prompt(self, max_results: int = 25, max_lessons: int = 5,
                          max_instructions: int = 5) -> str:
        """Résumé BORNÉ de la mémoire partagée, à insérer dans le prompt du LLM
        (même idée que l'index MEMORY.md de Claude Code : court, toujours chargé)."""
        lines = ["=== SHARED MEMORY (all agents of this campaign) ==="]
        goal = self.get_goal()
        if goal:
            lines.append(f"Goal: {json.dumps(goal)}")
        results = self.r.zrevrange(self.k("ranking"), 0, max_results - 1, withscores=True)
        total = self.r.zcard(self.k("ranking"))
        if results:
            lines.append(f"Molecules already simulated by the team ({total} total, best first):")
            lines += [f"  {m}\t{e:.3f}" for m, e in results]
            if total > max_results:
                lines.append(f"  ... and {total - max_results} more (lower energy).")
        running = {m: a for m, a in self.in_progress().items() if a != self.agent_id}
        if running:
            lines.append("Molecules being simulated RIGHT NOW by other agents: " + " ".join(sorted(running)))
        failed = sorted(self.failed_molecules())
        if failed:
            lines.append("Molecules whose simulation FAILED (invalid SMILES?): " + " ".join(failed[:20]))
        lessons = self.pull_lessons(count=max_lessons, exclude_own=True)
        if lessons:
            lines.append("Recent lessons from OTHER agents:")
            lines += [f"  - [{l['agent']}] {' '.join(l['text'].split())[:300]}" for l in lessons]
        instructions = self.pull_instructions(count=max_instructions)
        if instructions:
            lines.append("Instructions from the analysis agents (follow them):")
            lines += [f"  - ({i['status']}) {i['text']}" for i in instructions]
        lines.append("Do NOT propose molecules that are already listed above.")
        return "\n".join(lines)

    def in_progress(self) -> dict[str, str]:
        """Molécules en cours de calcul (verrou posé) → agent qui les calcule."""
        prefix = self.k("claim", "")
        out = {}
        for key in self.r.scan_iter(match=prefix + "*", count=200):
            owner = self.r.get(key)
            if owner is not None:
                out[key[len(prefix):]] = owner
        return out

    def known_molecules(self) -> list[str]:
        """Toutes les molécules à NE PAS reproposer : réussies, échouées, en cours."""
        return sorted(set(self.pull_results()) | self.failed_molecules() | set(self.in_progress()))

    # ------------------------------------------------------------------ tout en un
    async def get_or_compute(self, smiles: str,
                             compute: Callable[[str | None], Awaitable[float]],
                             new_task_id: Callable[[], str] | None = None,
                             poll: float = 2.0) -> float:
        """Renvoie l'énergie d'une molécule en évitant les calculs en double :

        1. déjà en mémoire            → on réutilise le résultat (ou l'échec) ;
        2. un autre agent la calcule  → on attend son résultat ;
        3. sinon on prend le verrou, on calcule, on publie le résultat, on libère.

        `compute(task_id)` lance le vrai calcul (xTB). Si le verrou d'un autre agent
        expire sans résultat (agent planté), on reprend le calcul nous-mêmes.
        """
        while True:
            known = self.get_result(smiles)
            if known is not None:
                self._count("reused_error" if known["status"] != "ok" else "reused")
                if known["status"] != "ok":
                    raise SimulationFailed(known.get("error") or "simulation failed before")
                return known["energy"]

            if self.claim(smiles):
                task_id = new_task_id() if new_task_id else None
                try:
                    energy = await compute(task_id)
                except Exception as exc:
                    # l'échec est mémorisé : les autres agents ne retenteront pas cette molécule
                    self.push_result(smiles, None, task_id=task_id, error=str(exc))
                    self.release(smiles)
                    self._count("computed_error")
                    raise
                except BaseException:      # annulation : on libère sans rien mémoriser
                    self.release(smiles)
                    raise
                # le résultat est publié AVANT de libérer le verrou : aucun autre agent
                # ne peut voir « ni verrou ni résultat » et relancer le calcul
                self.push_result(smiles, energy, task_id=task_id)
                self.release(smiles)
                self._count("computed")
                return energy

            # un autre agent tient le verrou : on attend son résultat
            self._count("waited")
            while self.claimed_by(smiles) is not None and self.get_result(smiles) is None:
                await asyncio.sleep(poll)
            # boucle : soit le résultat est là, soit le verrou a expiré → on retente

    # ------------------------------------------------------------------ résumé texte
    def summary(self) -> str:
        stats = self.r.hgetall(self.k("stats"))
        lines = [f"Mémoire partagée — campagne {self.campaign_id}",
                 f"  molécules réussies : {self.r.zcard(self.k('ranking'))}",
                 f"  molécules en échec : {self.r.scard(self.k('failed'))}",
                 f"  leçons             : {self.r.xlen(self.k('lessons'))}",
                 f"  instructions       : {self.r.xlen(self.k('instructions'))}",
                 "  compteurs          : " + (", ".join(f"{k}={v}" for k, v in sorted(stats.items())) or "aucun")]
        best = self.best(5)
        if best:
            lines.append("  5 meilleures       : " + ", ".join(f"{m} ({e:.3f})" for m, e in best))
        return "\n".join(lines)


# ---------------------------------------------------------------------- ligne de commande
def _cli(argv: list[str]) -> int:
    import redis
    r = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"),
                    port=int(os.environ.get("REDIS_PORT", "6379")), decode_responses=True)
    cmd = argv[0] if argv else "show"
    campaigns = r.zrange("mem:campaigns", 0, -1, withscores=True)
    if cmd == "campaigns":
        for c, t in campaigns:
            print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(t))}  {c}")
        return 0
    if cmd == "show":
        if len(argv) > 1:
            campaign = argv[1]
        elif campaigns:
            campaign = campaigns[-1][0]
        else:
            print("Aucune campagne en mémoire.")
            return 1
        mem = SharedMemory(r, campaign_id=campaign, agent_id="cli")
        print(mem.summary())
        print()
        print(mem.render_for_prompt())
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
