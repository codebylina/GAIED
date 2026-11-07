from __future__ import annotations

import asyncio
import logging
import os

from mol_design_agents import XTBConfig
from mol_design_agents import XTBSimulationAgent
import parsl
from parsl import Config
from parsl import HighThroughputExecutor
from parsl.concurrent import ParslPoolExecutor
from parsl.providers import LocalProvider

from academy.exchange import RedisExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

from flowcept import Flowcept
from shared_memory import SharedMemory  # [shared-memory]

logger = logging.getLogger(__name__)


async def main() -> int:
    init_logging(logging.INFO)

    config = Config(
        executors=[
            HighThroughputExecutor(
                provider=LocalProvider(
                    worker_init=(
                        ''  # [gaied-docker patch] no conda activate needed in Docker
                    ),
                ),
                max_workers_per_node=2,
                # [gaied-docker patch] a local LLM on CPU can starve the Parsl manager
                # for minutes; the default (120 s) made Parsl drop both agents (ManagerLost)
                heartbeat_threshold=int(os.environ.get('PARSL_HEARTBEAT_THRESHOLD', '1200')),
            ),
        ],
    )
    executor = ParslPoolExecutor(config)

    async with await Manager.from_exchange_factory(
        factory=RedisExchangeFactory(os.environ.get('REDIS_HOST', 'localhost'), 6379),  # [gaied-docker patch]
        executors=executor,
    ) as manager:
        seeds = os.environ.get('SEEDS', 'CNC(N)=O,CC1=C(O)N=C(O)N1').split(',')  # [gaied-docker patch]

        # [shared-memory] the Goal is written to the shared memory BEFORE the agents start
        memory = SharedMemory.from_env(agent_id='orchestrator')
        if memory is not None:
            if memory.get_goal() is None:
                memory.set_goal({
                    'objective': 'find molecules with the highest ionization energy (xTB)',
                    'seeds': seeds,
                    'constraints': ['valid SMILES', 'synthesizable', 'low cost'],
                })
            print(f'Shared memory ENABLED, campaign {memory.campaign_id}')
        else:
            print('Shared memory disabled (SHARED_MEMORY=false or Redis unreachable)')

        agents = []
        for molecule in seeds:
            agents.append(
                await manager.launch(
                    XTBSimulationAgent,
                    args=(
                        XTBConfig(),
                        molecule,
                        int(os.environ.get('MAX_ITERATIONS', '20')),   # max_iterations  # [gaied-docker patch]
                        3,   # no_improvement_tolerance
                    ),
                ),
            )

        print('Starting discovery campaign')
        print('=' * 80)
        # Track each agent's last known molecule count; exit when all stable.
        active: dict[int, tuple] = {i: (agent, -1, 0) for i, agent in enumerate(agents)}  # [gaied-docker stop-rule]
        max_stable = int(os.environ.get('STABLE_REPORTS', '20'))
        try:
            while active:
                await asyncio.sleep(30)
                finished = []
                for i, (agent, last_count, stable) in list(active.items()):
                    try:
                        report = await asyncio.wait_for(agent.report(), timeout=15.0)
                        count = len(report)
                        print(f'Progress report from agent {i}')
                        print(f'Number of molecules simulated: {count}')
                        print(f'Five best molecules: {report[:5]}')
                        stable = stable + 1 if (count == last_count and count > 0) else 0
                        if stable >= max_stable:
                            print(f'Agent {i} finished (no new molecules for {stable} reports).')
                            finished.append(i)
                        else:
                            active[i] = (agent, count, stable)
                    except Exception:
                        print(f'Agent {i} has finished.')
                        finished.append(i)
                for i in finished:
                    del active[i]
                print('=' * 80)
            print('All agents completed.')
        except (KeyboardInterrupt, asyncio.CancelledError):
            print('\nShutting down agents...')
        if memory is not None:  # [shared-memory] summary: reuses and avoided duplicates
            print(memory.summary())

    return 0


def _print_perf_summary(perf_csv: str) -> None:
    """Read the shared perf CSV and print a per-category aggregate table."""
    import csv as _csv
    from collections import defaultdict
    try:
        if not os.path.exists(perf_csv):
            return
        counts: dict = defaultdict(int)
        totals: dict = defaultdict(float)
        with open(perf_csv, newline='') as fh:
            for row in _csv.DictReader(fh):
                cat = row['category']
                counts[cat] += 1
                totals[cat] += float(row['elapsed_us'])
        if not counts:
            return
        col = 22
        header = (
            f"\n{'Category':<{col}} {'N':>7} {'Total(ms)':>11} {'Mean(µs)':>9}"
        )
        print('\n' + '=' * 60)
        print('Provenance overhead — all workers (from perf CSV):')
        print('=' * 60)
        print(header)
        print('-' * len(header.strip()))
        for cat in sorted(counts):
            n = counts[cat]
            total_us = totals[cat]
            mean_us = total_us / n if n else 0.0
            print(f"{cat:<{col}} {n:>7} {total_us/1000:>11.3f} {mean_us:>9.1f}")
    except Exception as exc:
        print(f'[perf summary] could not read {perf_csv}: {exc}')


if __name__ == '__main__':
    import signal
    import threading

    perf_csv = 'provenance_perf.csv'
    exit_code = 1
    with Flowcept() as flowcept:
        os.environ['FLOWCEPT_CAMPAIGN_ID'] = flowcept.campaign_id
        import flowcept.agents.academy.academy_plugin as _ap
        perf_csv = _ap._PERF_CSV_PATH or f'provenance_perf_{flowcept.campaign_id[:8]}.csv'
        os.environ['FLOWCEPT_PERF_CSV'] = perf_csv
        try:
            exit_code = asyncio.run(main())
        finally:
            # Suppress further Ctrl+C so cleanup isn't interrupted.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Run Parsl cleanup in a daemon thread; if it hangs, we still exit.
            def _parsl_cleanup():
                try:
                    parsl.dfk().cleanup()
                except Exception:
                    pass
            t = threading.Thread(target=_parsl_cleanup, daemon=True)
            t.start()
            t.join(timeout=15)
    _print_perf_summary(perf_csv)
    # os._exit bypasses atexit handlers (avoids the "DFK still running" warning).
    os._exit(exit_code)