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


class ScriptedLLM:
    def __init__(self, env, seed: int = 0) -> None:
        self.env = env
        self.rng = random.Random(seed)
        self.lock = threading.Lock()

    def __call__(self, system: str, user: str, purpose: str) -> str:
        with self.lock:
            return self._answer(system, user, purpose)

    def _answer(self, system, user, purpose):
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
