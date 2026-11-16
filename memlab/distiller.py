"""
L0 -> L1: the LLM proposes claims (atoms) from the raw message journal.

Batched: one LLM call for the new messages that fit an answer. Each claim keeps its source (the message
id, its author and the time it was written: the claim time used by the verifier). A claim that says the
same as a known atom is recorded as a restatement of that atom, not as a new atom (a restatement is not
evidence: evidence comes from the experiment registry). The LLM never decides whether a claim is true.
"""
from __future__ import annotations

import re
import time
import uuid

from .llm import LLM, extract_json, long_output_tokens

DISTILL_SYSTEM = """You turn the messages of a team of research agents into claims for the team's shared memory.
For EACH message below, write a section that starts with a line "### <message id>", then one JSON line:
{"claims": [{"text": "...", "same_as": null}]}
A claim is one self-contained statement that measurements could confirm or refute, or an advice for the team:
an effect of an option or a feature on the outcome, a comparison, a prediction about designs, or the outcome
the message reports for a design. Keep the message's meaning: do not add, strengthen, generalize or invert.
Do not turn plans, questions or intentions into claims. When a claim says the same as one of the KNOWN CLAIMS,
put that claim's id in "same_as" instead of writing a new one. A message without claims gets {"claims": []}."""


def norm(text: str) -> str:
    return " ".join(str(text).lower().strip(" .-").split())


class Distiller:
    def __init__(self, store, llm: LLM | None, on_atom=None, clock=time.time, tokens_per_message: int = 300,
                 known_chars: int = 3000) -> None:
        """on_atom(atom): called for every new atom (the verifier registers it)."""
        self.store, self.llm, self.on_atom, self.clock = store, llm, on_atom, clock
        self.tokens_per_message = tokens_per_message    # sizes batches from the answer budget
        self.known_chars = known_chars                  # how much of the known claims the prompt lists

    def step(self) -> int:
        if self.llm is None:
            return 0
        start = int((self.store.get("meta", "distill_cursor") or {}).get("i", 0))
        new = self.store.read_log("journal", start)
        if not new:
            return 0
        size = max(1, long_output_tokens() // max(1, self.tokens_per_message))
        batch = new[:size]
        self._distill(batch)
        self.store.put("meta", "distill_cursor", {"i": start + len(batch)})
        return len(batch)

    def _known(self) -> list[dict]:
        atoms = sorted(self.store.all("atoms"), key=lambda a: -a["created_at"])
        out, used = [], 0
        for a in atoms:
            used += len(a["text"]) + 20
            if used > self.known_chars:
                break
            out.append(a)
        return out

    def _distill(self, msgs: list[dict]) -> None:
        known = self._known()
        user = ("KNOWN CLAIMS:\n" + ("\n".join(f"[{a['id']}] {a['text']}" for a in known) or "(none)") +
                "\n\nMESSAGES:\n" + "\n".join(f"### {m['id']} (by {m['agent']}, {m['node']})\n{m['text']}" for m in msgs))
        answer = self.llm.complete(DISTILL_SYSTEM, user, purpose="memory", max_tokens=long_output_tokens())
        ids = {a["id"] for a in self.store.all("atoms")}
        by_text = {norm(a["text"]): a["id"] for a in self.store.all("atoms")}
        for m in msgs:
            sec = re.search(r"###\s*" + re.escape(m["id"]) + r"\b(.*?)(?=\n###\s|\Z)", answer, re.DOTALL)
            claims = (extract_json(sec.group(1)) or {}).get("claims", []) if sec else []
            self.store.append("distill_journal", {"t": self.clock(), "message": m["id"], "found": bool(sec),
                                                  "claims": claims if isinstance(claims, list) else []})
            for c in claims if isinstance(claims, list) else []:
                if not isinstance(c, dict) or not str(c.get("text", "")).strip():
                    continue
                same = str(c.get("same_as") or "")
                if same not in ids:
                    same = by_text.get(norm(c["text"]), "")   # the same words: a restatement whatever the LLM said
                if same in ids:
                    a = self.store.get("atoms", same)
                    a["restatements"].append({"message": m["id"], "agent": m["agent"], "t": m["t"]})
                    self.store.put("atoms", same, a)
                    continue
                atom = {"id": f"a-{uuid.uuid4().hex[:10]}", "text": str(c["text"]).strip(), "author": m["agent"],
                        "claimed_at": m["t"], "sources": [m["id"]], "restatements": [], "created_at": self.clock()}
                self.store.put("atoms", atom["id"], atom)
                ids.add(atom["id"])
                by_text[norm(atom["text"])] = atom["id"]
                if self.on_atom:
                    self.on_atom(atom)
