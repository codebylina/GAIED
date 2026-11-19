# Development campaigns

Campaigns run during development, before the evaluation version was frozen (Qwen 3.5 9B through Ollama, real xTB,
two agents, three iterations). The code changed between them, so they are not part of the evaluation. Each
observation is classified as handled automatically by the existing rules, not covered (a general problem type), or a
bug that was fixed.

| Campaign | Handled automatically | Not covered | Bug fixed |
|---|---|---|---|
| real1 (two seeds; cut short by the stop rule) | guidance delivered; an agent ignored "do not propose these invalid SMILES again", recorded as not applied; M3 filled by the agents with kinds and statuses | | proposals compared with the memory too late (false "already measured"); list-typed procedure fields |
| real2 (cut short) | genuine repeated proposals detected; invalid-SMILES failures diagnosed; a wrong LLM explanation kept as `candidate_explanation`; M4 procedure derived from two experiences | the same failure diagnosed twice (memory and provenance) | formatting of procedure actions |
| real3 (complete) | reuse of another agent's result; full guidance cycle (6 applied, 5 effective, 1 not effective); contradicting evidence added without overwriting; triple computation detected; repeated proposals stopped after targeted guidance | simultaneous duplicate computations (lock added); effectiveness judged once and agent-declared epistemic status (documented limitations) | evidence of knowledge about a failed molecule; misleading task-error text; guidance counting units; unusable LLM answers not logged; "applied" test unsuited to invalid-SMILES guidance (RDKit test); procedure never drafted again after an unusable LLM answer |
| lock1 (same seed twice; namespace reused) | iterations without a tool call end instead of blocking the agent; invalid-SMILES guidance judged by RDKit; M4 procedure created | a molecule whose failure is confirmed can be proposed and computed again in a later iteration (documented limitation) | an agent blocked for 12 minutes (unbounded tool-call retries in the original loop); an iteration without proposals counted as applied guidance; M4 status counting deliveries as applications |
| lock2 (same seed twice) | one computation for a molecule proposed by both agents at once; guidance not applied, then applied and effective; no procedure without repeated failures | evidence cited by an agent for a wrong statement (documented limitation) | |

Operational issues, unrelated to the system: Ollama stopped (real2, first try); a too short stop rule given in the
commands (real1, real2).
