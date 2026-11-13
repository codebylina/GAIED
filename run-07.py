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
from evident import EvidentMemory  # [evident]
from evident.monitor import start_in_background as start_monitor  # [evident]

logger = logging.getLogger(__name__)


async def main() -> int:
    init_logging(logging.INFO)
    seeds = os.environ.get('SEEDS', 'CNC(N)=O,CC1=C(O)N=C(O)N1').split(',')  # [gaied-docker patch]

    config = Config(
        executors=[
            HighThroughputExecutor(
                provider=LocalProvider(
                    worker_init=(
                        ''  # [gaied-docker patch] no conda activate needed in Docker
                    ),
                ),
                # [evident] one Parsl worker per agent (was 2: a third agent never started)
                max_workers_per_node=int(os.environ.get('PARSL_MAX_WORKERS', max(2, len(seeds)))),
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
        # [evident] Goal and run configuration are written BEFORE the agents start
        memory = EvidentMemory.from_env(agent_id='orchestrator')
        tolerance = int(os.environ.get('NO_IMPROVEMENT_TOLERANCE', '3')) or None
        if memory is not None:
            if memory.get_goal() is None:
                memory.set_goal({
                    'objective': 'find molecules with the highest ionization energy (xTB)',
                    'seeds': seeds,
                    'constraints': {
                        'max_heavy_atoms': int(os.environ.get('GOAL_MAX_HEAVY_ATOMS', '12')),
                        'allowed_elements': os.environ.get('GOAL_ALLOWED_ELEMENTS', 'C,H,N,O,F').split(','),
                    },
                    'guidelines': ['valid SMILES', 'synthesizable', 'low cost'],
                })
            memory.set_meta(
                mode=memory.mode, seeds=seeds,
                monitor=os.environ.get('MONITOR', 'false').lower() == 'true',
                inject=os.environ.get('INJECT_FALSE_CLAIMS', 'false').lower() == 'true',
                max_iterations=int(os.environ.get('MAX_ITERATIONS', '20')),
                tolerance=tolerance, llm=os.environ.get('LLM_MODEL') or os.environ.get('OLLAMA_MODEL'),
                condition=os.environ.get('EXPERIMENT_CONDITION', ''),
            )
            print(f'EVIDENT memory: mode {memory.mode}, campaign {memory.campaign_id}')
        else:
            print('EVIDENT memory unavailable (Redis unreachable): agents run without memory')
        monitor_stop = asyncio.Event()
        monitor_task = start_monitor(monitor_stop)  # [evident] real-time analysis agent (MONITOR=true)

        agents = []
        for molecule in seeds:
            agents.append(
                await manager.launch(
                    XTBSimulationAgent,
                    args=(
                        XTBConfig(),
                        molecule,
                        int(os.environ.get('MAX_ITERATIONS', '20')),   # max_iterations  # [gaied-docker patch]
                        tolerance,   # no_improvement_tolerance  # [evident] NO_IMPROVEMENT_TOLERANCE, 0 = off
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
        monitor_stop.set()
        if monitor_task is not None:
            await asyncio.wait({monitor_task}, timeout=5)
        if memory is not None:  # [evident] summary, and export for the analysis
            print(memory.summary())
            export_path = os.environ.get('EVIDENT_EXPORT')
            if export_path:
                import json
                os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
                with open(export_path, 'w') as fh:
                    json.dump(memory.export(), fh)
                print(f'Campaign exported to {export_path}')

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