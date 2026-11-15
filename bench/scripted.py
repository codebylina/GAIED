"""
Scripted stand-in for the LLM, for dry runs and tests of the whole pipeline without a model.
It answers every prompt of the agents, the judges, the baselines and the oracle evaluator with
simple but well-formed outputs. Results obtained with it say nothing about the method.
"""
from __future__ import annotations

import json
import random
import re
import threading

from provmem.judge import FORMALIZE_SYSTEM, REVIEW_SKEPTIC, TEXT_SYSTEM, VERDICT_SYSTEM

CLAIM_RE = re.compile(r"Option ([A-Z])(\d+) (increases|decreases)")

TEST_CODE = """def test(df):
    ok = df[df['status'] == 'ok']
    col = 'slot_{s}'
    ref = ok[ok[col] != {o}]['value'].mean() if (ok[col] != {o}).any() else None
    inside = ok[ok[col] == {o}]
    if ref is None or len(inside) == 0:
        return {{'in_scope': [], 'support': [], 'contradict': [], 'stats': {{}}, 'note': 'no data'}}
    good = inside[inside['value'] {op} ref]
    bad = inside[~(inside['value'] {op} ref)]
    return {{'in_scope': list(ok['id']), 'support': list(good['id']), 'contradict': list(bad['id']),
            'stats': {{'n_in': int(len(inside)), 'ref_mean': float(ref), 'in_mean': float(inside['value'].mean())}},
            'note': 'each design with the option is compared with the mean of designs without it'}}
"""

HOLDS_CODE = """def holds(df):
    ok = df[df['status'] == 'ok']
    col = 'slot_{s}'
    a = ok[ok[col] == {o}]['value'].mean()
    b = ok[ok[col] != {o}]['value'].mean()
    return bool(a {op} b)
"""


class ScriptedLLM:
    def __init__(self, env, seed: int = 0) -> None:
        self.env = env
        self.rng = random.Random(seed)
        self.lock = threading.Lock()

    def __call__(self, system: str, user: str, purpose: str) -> str:
        with self.lock:
            return self._answer(system, user, purpose)

    def _answer(self, system, user, purpose):
        if system == FORMALIZE_SYSTEM:
            m = CLAIM_RE.search(user.split("CLAIM:")[1].split("\n")[0] if "CLAIM:" in user else user)
            if not m:
                return '{"testable": false, "scoped_statement": "n/a", "relation": {"type": "new", "target": null}}'
            s, o, d = m.group(1), m.group(2), m.group(3)
            code = TEST_CODE.format(s=s, o=o, op=">" if d == "increases" else "<")
            return json.dumps({"testable": True, "scoped_statement": m.group(0) + " the outcome",
                               "relation": {"type": "new", "target": None}}) + f"\n```python\n{code}```"
        if system == VERDICT_SYSTEM or system.startswith(VERDICT_SYSTEM[:40]):
            c = re.search(r"support=(\d+) contradict=(\d+)", user)
            ns, nc = (int(c.group(1)), int(c.group(2))) if c else (0, 0)
            rows = re.findall(r"^(\S+) \| .* \| (support|contradict)$", user, re.M)
            sup = [i for i, r in rows if r == "support"]
            con = [i for i, r in rows if r == "contradict"]
            v = "supported" if ns > nc and ns >= 2 else "contradicted" if nc > ns and nc >= 2 else "inconclusive"
            return json.dumps({"verdict": v, "cite_support": sup if v == "supported" else [],
                               "cite_contradict": con if v == "contradicted" else [], "rationale": f"{ns} for, {nc} against"})
        if system == REVIEW_SKEPTIC:
            return "The sample may be small."
        if system == TEXT_SYSTEM:
            return '{"verdict": "supported", "relation": {"type": "new", "target": null}, "rationale": "plausible"}'
        if purpose == "memory":
            if "facts" in system:
                return json.dumps({"facts": [l.strip("- ").strip() for l in user.splitlines() if CLAIM_RE.search(l)]})
            return '{"op": "ADD", "id": null, "text": ""}'
        if purpose == "oracle":
            m = CLAIM_RE.search(user)
            if not m:
                return '{"testable": false}'
            s, o, d = m.group(1), m.group(2), m.group(3)
            return '{"testable": true}\n```python\n' + HOLDS_CODE.format(s=s, o=o, op=">" if d == "increases" else "<") + "```"
        # agent prompts
        if "Choose exactly" in user:
            n = int(re.search(r"Choose exactly (\d+)", user).group(1))
            own = re.findall(r"^((?:[A-Z]\d+-)+[A-Z]\d+): (-?[\d.eE+-]+)$", user, re.M)
            base = max(own, key=lambda x: float(x[1]))[0] if own else self.env.seeds(1)[0]
            combo = list(self.env.parse(base))
            props = []
            for _ in range(n):
                c = combo[:]
                i = self.rng.randrange(len(c))
                c[i] = self.rng.randrange(self.env.spec.n_options)
                props.append(self.env.key(tuple(c)))
            return json.dumps({"proposals": props})
        if "Write your new conclusions" in user:
            own = re.findall(r"^((?:[A-Z]\d+-)+[A-Z]\d+): (-?[\d.eE+-]+)$", user, re.M)
            if len(own) < 2:
                return "- Too few results to conclude."
            best = self.env.parse(max(own, key=lambda x: float(x[1]))[0])
            i = self.rng.randrange(len(best))
            return f"- Option {self.env.slots[i]}{best[i]} increases the outcome compared with the other options of slot {self.env.slots[i]}."
        return "Plan: vary one slot at a time around the best design found so far."
