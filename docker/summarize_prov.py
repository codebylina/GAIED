"""
summarize_prov.py — Résume la provenance Flowcept d'une campagne GAIED.

Affiche, dans l'ordre où Flowcept les a enregistrés :
  - chaque nœud LangGraph (plan, tool_calling, simulate, conclude, critique, update) et sa durée ;
  - chaque appel au LLM : durée, longueur de la réponse, nombre de molécules proposées ;
  - chaque simulation xTB : molécule, énergie ou erreur.
Puis un résumé : temps total passé dans le LLM, nombre de simulations, etc., et le nombre
de calculs xTB faits EN DOUBLE (même molécule calculée plusieurs fois) : c'est la mesure
qui compare une campagne avec et sans mémoire partagée.

Usage (depuis GAIED/docker) :
    docker compose run --rm gaied python docker/summarize_prov.py            # fichier le plus récent
    docker compose run --rm gaied python docker/summarize_prov.py output/flowcept_buffer_XXX.jsonl
"""
import glob
import json
import os
import sys
from collections import Counter

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    Chem = None


def _canonical(smiles):
    mol = Chem.MolFromSmiles(smiles) if Chem is not None else None
    return Chem.MolToSmiles(mol) if mol is not None else str(smiles).strip()


def _short(value, n=70):
    text = " ".join(str(value).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            import ast
            v = ast.literal_eval(value)
            return v if isinstance(v, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def main(path: str | None) -> int:
    if path is None:
        files = glob.glob("output/flowcept_buffer_*.jsonl")
        if not files:
            print("Aucun fichier output/flowcept_buffer_*.jsonl trouvé.")
            return 1
        path = max(files, key=os.path.getmtime)
    print(f"Fichier : {path}\n")

    records = [json.loads(line) for line in open(path) if line.strip()]
    llm_total_s, llm_calls, empty_tool_calls, sims, sim_errors = 0.0, 0, 0, 0, 0
    xtb_runs = Counter()   # molécule canonique → nombre de vrais calculs xTB

    for r in records:
        st = r.get("subtype")
        used, gen = r.get("used") or {}, r.get("generated") or {}
        dur = None
        if r.get("started_at") and r.get("ended_at"):
            dur = r["ended_at"] - r["started_at"]

        if st == "langgraph_node":
            name = (r.get("custom_metadata") or {}).get("node_name") or r.get("activity_id")
            print(f"[nœud]  {name:<13} {'' if dur is None else f'{dur:7.1f} s'}")

        elif st == "llm_call":
            llm_calls += 1
            secs = ((r.get("custom_metadata") or {}).get("elapsed_ms") or 0) / 1000
            llm_total_s += secs
            response = gen.get("response") or ""
            calls = gen.get("tool_calls") or []
            line = f"[LLM]   {secs:7.1f} s  réponse de {len(response):5d} caractères"
            if used.get("has_tools"):
                line += f"  → {len(calls)} molécule(s) proposée(s)"
                if not calls:
                    empty_tool_calls += 1
                    line += "  ⚠ AUCUN APPEL D'OUTIL : « " + _short(response, 60) + " »"
            print(line)

        elif st in ("tool_call", "parsl_task"):
            sims += 1
            inp = _as_dict(used.get("input")) or used
            smiles = inp.get("smiles") or used.get("smiles") or _short(used, 40)
            out = gen.get("output", gen.get("ionization_energy")) if isinstance(gen, dict) else gen
            status = r.get("status")
            if status == "ERROR" or out is None:
                sim_errors += 1
                out = "ERREUR " + _short(r.get("stderr") or gen, 50)
            print(f"[xTB]   {smiles:<28} → {out}  {'' if dur is None else f'({dur:.1f} s)'}")
            if st == "parsl_task":          # un vrai calcul xTB (pas une simple réutilisation)
                xtb_runs[_canonical(smiles)] += 1

    print("\n" + "=" * 60)
    print(f"Appels au LLM            : {llm_calls}, au total {llm_total_s / 60:.1f} min")
    if llm_calls:
        print(f"Durée moyenne d'un appel : {llm_total_s / llm_calls:.0f} s")
    print(f"Appels sans molécule     : {empty_tool_calls}  (le modèle n'a pas produit le JSON attendu)")
    print(f"Simulations xTB          : {sims}, dont {sim_errors} en erreur")
    if xtb_runs:
        dup = {m: n for m, n in xtb_runs.items() if n > 1}
        wasted = sum(n - 1 for n in dup.values())
        print(f"Calculs xTB réels        : {sum(xtb_runs.values())} pour {len(xtb_runs)} molécules différentes")
        print(f"Calculs en DOUBLE        : {wasted}" + (
            "  (" + ", ".join(f"{m} ×{n}" for m, n in sorted(dup.items(), key=lambda x: -x[1])) + ")" if dup else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
