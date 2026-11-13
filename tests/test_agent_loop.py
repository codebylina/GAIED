"""
The real agent graph with an LLM that NEVER returns a parsable tool call: the agent must
finish its iterations instead of staying blocked in tool_calling (bounded retries), and
the failures must be recorded in the telemetry. No real LLM, fake xTB, a few seconds.

    docker compose run --rm --no-deps gaied python tests/test_agent_loop.py
"""
import asyncio
import json
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "docker")]

from experiments.synthetic import ScriptedLLM, _Resp, fake_energy  # noqa: E402  (also stubs xTB if missing)
import mol_design_agents as mda  # noqa: E402
from evident import EvidentMemory  # noqa: E402


class NoToolCallLLM(ScriptedLLM):
    """Answers normally, except that tool_calling never gets a parsable JSON."""

    def bind_tools(self, tools):
        b = NoToolCallLLM(self.model, self.temperature)
        b._tools = tools
        return b

    async def ainvoke(self, messages):
        if self._tools:
            self.calls = getattr(self, "calls", 0) + 1
            return _Resp('I would like to simulate {"tool_calls": [{"name": "compute_ioniz')
        return await super().ainvoke(messages)


def test_agent_does_not_block_without_tool_calls():
    campaign = f"test-loop-{uuid.uuid4().hex[:8]}"
    os.environ.update({"MEMORY_CAMPAIGN_ID": campaign, "MEMORY_MODE": "evident",
                       "N_XTB_WORKERS": "1", "TOOL_CALL_RETRIES": "2",
                       "SYNTH_LLM_SECONDS": "0", "SYNTH_XTB_SECONDS": "0"})
    os.environ.pop("FLOWCEPT_CAMPAIGN_ID", None)
    mda.SomeAPILLM = NoToolCallLLM
    mda._compute_energy_sync = fake_energy

    async def main():
        agent = mda.XTBSimulationAgent(mda.XTBConfig(), "CNC(N)=O", 2, None)
        await agent.agent_on_startup()
        try:
            await asyncio.wait_for(agent.conduct_simulation_campaign(asyncio.Event()), timeout=60)
        finally:
            agent.pool.shutdown()
        return agent

    asyncio.run(main())
    mem = EvidentMemory.from_env("cli")
    events = [f for _, f in mem.r.xrange(mem.k("telemetry"))]
    failures = [json.loads(f["d"]) for f in events if f["e"] == "tool_call_failed"]
    assert len(failures) == 2, f"one failure per iteration expected, got {len(failures)}"
    assert failures[0]["attempts"] == 3 and "tool_calls" in failures[0]["responses"][0]
    assert sum(1 for f in events if f["e"] == "iteration_end") == 2, "both iterations completed"
    for key in mem.r.scan_iter(f"ev:{campaign}:*"):
        mem.r.delete(key)
    print(f"    {len(failures)} iterations without tool call, agent finished normally")


if __name__ == "__main__":
    try:
        test_agent_does_not_block_without_tool_calls()
        print("[OK]    test_agent_does_not_block_without_tool_calls\n\n1/1 tests passed")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[FAIL]  test_agent_does_not_block_without_tool_calls: {exc!r}\n\n0/1 tests passed")
        sys.exit(1)
