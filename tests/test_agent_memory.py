"""
Test d'intégration : la mémoire partagée branchée dans le VRAI code de l'agent
(XTBSimulationAgent.compute_ionization_energy), avec un vrai calcul xTB mais SANS LLM.
Durée : moins d'une minute.

Dans Docker, depuis GAIED/docker :
    docker compose run --rm gaied python tests/test_agent_memory.py

Ce que l'on vérifie :
  1. deux agents demandent EN MÊME TEMPS la même molécule, écrite de deux façons
     (CCO et OCC) → xTB ne tourne qu'UNE fois, les deux obtiennent la même énergie ;
  2. un troisième agent, lancé après, réutilise le résultat sans calculer ;
  3. une molécule invalide échoue une fois, puis l'échec est réutilisé ;
  4. le résultat en mémoire porte le task_id envoyé à Flowcept (lien mémoire ↔ provenance) ;
  5. avec la mémoire désactivée, l'agent fonctionne comme avant (et recalcule).

Options (pour vérifier sans Docker) : --fake (Redis simulé), --fake-xtb (xTB simulé).
"""
import asyncio
import os
import sys
import time
import types
import uuid
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docker"))

if "--fake-xtb" in sys.argv:                      # xTB simulé (machine sans xTB)
    fake = types.ModuleType("xtb.ase.calculator")
    fake.XTB = object
    sys.modules.update({"xtb": types.ModuleType("xtb"), "xtb.ase": types.ModuleType("xtb.ase"),
                        "xtb.ase.calculator": fake})

import mol_design_agents as mda                   # noqa: E402
from shared_memory import SharedMemory, SimulationFailed, canonical_smiles  # noqa: E402

if "--fake-xtb" in sys.argv:
    def _fake_energy(config, smiles):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Parse failure for {smiles}")
        time.sleep(1.0)
        return 10.0 + mol.GetNumAtoms() / 10
    mda._compute_energy_sync = _fake_energy        # hérité par les processus xTB (fork)


def make_client():
    if "--fake" in sys.argv:
        import fakeredis
        return fakeredis.FakeRedis(decode_responses=True)
    import redis
    client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)
    client.ping()
    return client


CLIENT = make_client()
CAMPAIGN = f"test-agent-{uuid.uuid4().hex[:8]}"
POOL = ProcessPoolExecutor(max_workers=2)


def make_agent(name, with_memory=True):
    agent = mda.XTBSimulationAgent(mda.XTBConfig(), "CNC(N)=O")
    agent._academy_interceptor = None              # pas de Flowcept dans ce test
    agent.pool = POOL
    agent.agent_name = name
    agent.memory = SharedMemory(CLIENT, campaign_id=CAMPAIGN, agent_id=name) if with_memory else None
    return agent


def stats():
    return {k: int(v) for k, v in CLIENT.hgetall(f"mem:{CAMPAIGN}:stats").items()}


def test_concurrent_agents_compute_once():
    a, b = make_agent("A"), make_agent("B")

    async def main():
        return await asyncio.gather(a.compute_ionization_energy("CCO"),
                                    b.compute_ionization_energy("OCC"))

    ea, eb = asyncio.run(main())
    s = stats()
    assert ea == eb, (ea, eb)
    assert s.get("computed") == 1, f"xTB doit tourner une seule fois : {s}"
    assert s.get("waited", 0) + s.get("reused", 0) >= 1, s
    print(f"    énergie de l'éthanol = {ea:.3f} ; compteurs = {s}")


def test_late_agent_reuses():
    before = stats().get("computed", 0)
    t0 = time.time()
    e = asyncio.run(make_agent("C").compute_ionization_energy("C(O)C"))
    assert stats().get("computed", 0) == before, "aucun nouveau calcul attendu"
    print(f"    réutilisé en {time.time() - t0:.3f} s (énergie {e:.3f})")


def test_invalid_molecule_failure_is_shared():
    a, b = make_agent("A"), make_agent("B")
    try:
        asyncio.run(a.compute_ionization_energy("C1CC(("))
        raise AssertionError("une molécule invalide doit échouer")
    except SimulationFailed:
        raise AssertionError("A doit avoir la vraie erreur, pas un échec mémorisé")
    except Exception:
        pass
    try:
        asyncio.run(b.compute_ionization_energy("C1CC(("))
        raise AssertionError("B doit recevoir l'échec")
    except SimulationFailed:
        pass
    assert stats().get("computed_error") == 1 and stats().get("reused_error") == 1, stats()


def test_result_carries_flowcept_task_id():
    rec = SharedMemory(CLIENT, campaign_id=CAMPAIGN, agent_id="X").get_result("CCO")
    assert rec and rec.get("task_id"), rec
    uuid.UUID(rec["task_id"])                      # c'est bien un identifiant de tâche
    print(f"    CCO calculé par {rec['agent']}, tâche Flowcept {rec['task_id']}")


def test_without_memory_still_works():
    a, b = make_agent("A", with_memory=False), make_agent("B", with_memory=False)

    async def main():
        return await asyncio.gather(a.compute_ionization_energy("CC"),
                                    b.compute_ionization_energy("CC"))

    ea, eb = asyncio.run(main())
    assert ea == eb and "CC" in a.molecule_cache and "CC" in b.molecule_cache
    assert CLIENT.hgetall(f"mem:{CAMPAIGN}:result:{canonical_smiles('CC')}") == {}


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    try:
        for t in tests:
            try:
                t()
                print(f"[OK]    {t.__name__}")
            except Exception as exc:
                failed += 1
                print(f"[ÉCHEC] {t.__name__} : {exc!r}")
    finally:
        for key in CLIENT.scan_iter(f"mem:{CAMPAIGN}:*"):
            CLIENT.delete(key)
        POOL.shutdown()
    print(f"\n{len(tests) - failed}/{len(tests)} tests réussis")
    sys.exit(1 if failed else 0)
