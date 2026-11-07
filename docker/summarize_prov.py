"""
summarize_prov.py — Summarizes the Flowcept provenance of a GAIED campaign.

Prints, in time order:
  - each LangGraph node (plan, tool_calling, simulate, conclude, critique, update) and its duration;
  - each LLM call: duration, response length, number of proposed molecules;
  - each xTB simulation: molecule, energy or error.
Then a summary: total time spent in the LLM, number of simulations, and the number of
DUPLICATE xTB computations (same molecule computed several times): this is the metric
that compares a campaign with and without shared memory.

Usage (from GAIED/docker):
    docker compose run --rm gaied python docker/summarize_prov.py            # latest campaign
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


def _read(path):
    records = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _campaign_of(records):
    for r in records:
        if r.get("campaign_id"):
            return r["campaign_id"]
    return None


def _load(path):
    """Flowcept writes ONE file per process (orchestrator + one per agent).
    Gather all the files that belong to the same campaign as the most recent one."""
    if path is not None:
        return [path], _read(path)
    files = sorted(glob.glob("output/flowcept_buffer_*.jsonl"), key=os.path.getmtime, reverse=True)
    by_file = {f: _read(f) for f in files}
    campaign = next((c for c in (_campaign_of(by_file[f]) for f in files) if c), None)
    if campaign is None:
        return [], []
    chosen = [f for f in files if _campaign_of(by_file[f]) == campaign]
    records = [r for f in chosen for r in by_file[f]]
    records.sort(key=lambda r: r.get("started_at") or r.get("utc_timestamp") or 0)
    return chosen, records


def main(path: str | None) -> int:
    files, records = _load(path)
    if not records:
        print("No provenance found in output/flowcept_buffer_*.jsonl.")
        return 1
    print(f"Campaign {_campaign_of(records)} - {len(files)} file(s):")
    for f in files:
        print(f"  {f}")
    print()

    llm_total_s, llm_calls, empty_tool_calls, sims, sim_errors = 0.0, 0, 0, 0, 0
    xtb_runs = Counter()   # canonical molecule -> number of actual xTB computations
    tool_calls = Counter()  # canonical molecule -> number of tool requests by the LLM

    for r in records:
        st = r.get("subtype")
        used, gen = r.get("used") or {}, r.get("generated") or {}
        dur = None
        if r.get("started_at") and r.get("ended_at"):
            dur = r["ended_at"] - r["started_at"]

        if st == "langgraph_node":
            name = (r.get("custom_metadata") or {}).get("node_name") or r.get("activity_id")
            print(f"[node]  {name:<13} {'' if dur is None else f'{dur:7.1f} s'}")

        elif st == "llm_call":
            llm_calls += 1
            secs = ((r.get("custom_metadata") or {}).get("elapsed_ms") or 0) / 1000
            llm_total_s += secs
            response = gen.get("response") or ""
            calls = gen.get("tool_calls") or []
            line = f"[LLM]   {secs:7.1f} s  response of {len(response):5d} characters"
            if used.get("has_tools"):
                line += f"  -> {len(calls)} molecule(s) proposed"
                if not calls:
                    empty_tool_calls += 1
                    line += "  WARNING NO TOOL CALL: \"" + _short(response, 60) + "\""
            print(line)

        elif st in ("tool_call", "parsl_task"):
            sims += 1
            inp = _as_dict(used.get("input")) or used
            smiles = inp.get("smiles") or used.get("smiles") or _short(used, 40)
            out = gen.get("output", gen.get("ionization_energy")) if isinstance(gen, dict) else gen
            status = r.get("status")
            if status == "ERROR" or out is None:
                sim_errors += 1
                out = "ERROR " + _short(r.get("stderr") or gen, 50)
            print(f"[{'xTB' if st == 'parsl_task' else 'tool'}]  {smiles:<28} -> {out}  {'' if dur is None else f'({dur:.1f} s)'}")
            if st == "parsl_task":          # an actual xTB computation (not a reuse)
                xtb_runs[_canonical(smiles)] += 1
            else:
                tool_calls[_canonical(smiles)] += 1

    print("\n" + "=" * 60)
    print(f"LLM calls                : {llm_calls}, {llm_total_s / 60:.1f} min in total")
    if llm_calls:
        print(f"Mean LLM call duration   : {llm_total_s / llm_calls:.0f} s")
    print(f"Calls without molecules  : {empty_tool_calls}  (the model did not produce the expected JSON)")
    print(f"Tool calls (simulations) : {sum(tool_calls.values())}")
    print(f"xTB tasks recorded       : {sum(xtb_runs.values())}, errors among all: {sim_errors}")
    if not xtb_runs and tool_calls:   # older campaigns: no parsl_task record
        print("(no record of the xTB computations themselves: counting tool requests instead)")
        xtb_runs = tool_calls
    if xtb_runs:
        dup = {m: n for m, n in xtb_runs.items() if n > 1}
        wasted = sum(n - 1 for n in dup.values())
        print(f"xTB computations         : {sum(xtb_runs.values())} for {len(xtb_runs)} distinct molecules")
        print(f"DUPLICATE computations   : {wasted}" + (
            "  (" + ", ".join(f"{m} ×{n}" for m, n in sorted(dup.items(), key=lambda x: -x[1])) + ")" if dup else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
