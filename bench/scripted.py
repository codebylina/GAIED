"""
Scripted stand-in for the LLM, for dry runs and tests of the whole pipeline without a model.
It answers every prompt of the agents, Mem0 and the oracle evaluator with
simple but well-formed outputs. Results obtained with it say nothing about the method.
"""
from __future__ import annotations

import json
import random
import re
import threading


CLAIM_RE = re.compile(r"Option ([A-Z])(\d+) (increases|decreases)")

HOLDS_CODE = """def holds(df):
    ok = df[df['status'] == 'ok']
    col = 'slot_{s}'
    a = ok[ok[col] == {o}]['value'].mean()
    b = ok[ok[col] != {o}]['value'].mean()
    return bool(a {op} b)
"""


def mem0_scripted(dims: int = 768):
    """Stand-ins for Mem0's LLM (memory extraction) and embedder."""
    import hashlib
    import math

    def llm(messages):
        user = messages[-1]["content"] if messages else ""
        new = user.split("New Messages")[-1] if "New Messages" in user else user
        facts = [l.strip(" -") for l in new.splitlines() if CLAIM_RE.search(l) or re.search(r"[A-Z]\d+-[A-Z]\d+", l)]
        return json.dumps({"memory": [{"text": f} for f in facts[:5]]})

    def embed(text):
        v = [0.0] * dims
        for w in re.findall(r"\w+", str(text).lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dims] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]
    return llm, embed


GEN_TEST = """def test(df):
    ok = df[df['status'] == 'ok']
    col = 'slot_{s}'
    inside = ok[ok[col] == {o}]
    ref = ok[ok[col] != {o}]['value'].mean() if (ok[col] != {o}).any() else None
    if ref is None or len(inside) == 0:
        return {{'in_scope': [], 'support': [], 'contradict': [], 'stats': {{}}, 'verdict': 'insufficient'}}
    good = inside[inside['value'] {op} ref]
    bad = inside[~(inside['value'] {op} ref)]
    v = 'supported' if len(good) >= 2 and len(good) > len(bad) else 'contradicted' if len(bad) >= 2 and len(bad) > len(good) else 'insufficient'
    return {{'in_scope': list(ok['id']), 'support': list(good['id']), 'contradict': list(bad['id']),
            'stats': {{'ref': float(ref)}}, 'verdict': v}}
"""

FACT_TEST = """def test(df):
    rows = df[df['entity'] == '{e}']
    ok = rows[rows['status'] == 'ok']
    good = ok[(ok['value'] - {v}).abs() <= 0.05 * max(1.0, abs({v}))]
    bad = ok[(ok['value'] - {v}).abs() > 0.05 * max(1.0, abs({v}))]
    v = 'supported' if len(good) else 'contradicted' if len(bad) else 'insufficient'
    return {{'in_scope': list(rows['id']), 'support': list(good['id']), 'contradict': list(bad['id']), 'stats': {{}}, 'verdict': v}}
"""


def scripted_distiller(user: str) -> str:
    """One claim per conclusion line that names an option effect; same_as when the text is already known."""
    known = dict((t.strip(), i) for i, t in re.findall(r"^\[(a-[0-9a-f]+)\] (.+)$", user, re.M))
    out = []
    for mid, body in re.findall(r"^### (\S+) \(by [^)]*\)\n(.*?)(?=^### |\Z)", user.split("MESSAGES:", 1)[-1], re.M | re.S):
        claims = []
        for line in body.splitlines():
            text = line.strip(" -")
            if CLAIM_RE.search(text):
                claims.append({"text": text, "same_as": known.get(text)})
        out.append(f"### {mid}\n" + json.dumps({"claims": claims}))
    return "\n".join(out)


def scripted_verifier(user: str, text_mode: bool) -> str:
    out = []
    for block in re.split(r"(?m)^### ", user.split("MEMORIES TO REVIEW:")[-1].split("MEMORIES:")[-1])[1:]:
        mid = block.split()[0]
        body = block[len(mid):]
        if text_mode:
            out.append(f'### {mid} {{"verdict": "true"}}')
            continue
        m = CLAIM_RE.search(body)
        f = re.search(r"((?:[A-Z]\d+-)+[A-Z]\d+)\D{0,20}?(-?\d+\.\d+)", body)
        if m:
            s_, o, d = m.group(1), m.group(2), m.group(3)
            code = GEN_TEST.format(s=s_, o=o, op=">" if d == "increases" else "<")
            out.append(f'### {mid}\n{{"kind": "generalization", "scoped_statement": "{m.group(0)} the outcome"}}\n```python\n{code}```')
        elif f:
            code = FACT_TEST.format(e=f.group(1), v=float(f.group(2)))
            out.append(f'### {mid}\n{{"kind": "report", "scoped_statement": "{f.group(1)} gave {f.group(2)}"}}\n```python\n{code}```')
        else:
            out.append(f'### {mid}\n{{"kind": "untestable", "scoped_statement": "n/a"}}')
    return "\n".join(out)


class ScriptedLLM:
    def __init__(self, env, seed: int = 0) -> None:
        self.env = env
        self.rng = random.Random(seed)
        self.lock = threading.Lock()

    def __call__(self, system: str, user: str, purpose: str) -> str:
        with self.lock:
            return self._answer(system, user, purpose)

    def _answer(self, system, user, purpose):
        from memlab.distiller import DISTILL_SYSTEM
        from memlab.verifier import FORMALIZE_SYSTEM as VF, TEXT_SYSTEM as VT
        if system in (VF, VT):
            return scripted_verifier(user, system == VT)
        if system == DISTILL_SYSTEM:
            return scripted_distiller(user)
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
