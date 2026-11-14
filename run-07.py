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
from falda_memory import FaldaMemory, Recorder  # [falda]

logger = logging.getLogger(__name__)


GOAL = {  # [falda] Goal and hard constraints, written as the Core of the FALDA pool
    'objective': 'find molecules with the highest ionization energy (xTB)',
    'constraints': {
        'max_heavy_atoms': int(os.environ.get('GOAL_MAX_HEAVY_ATOMS', '12')),
        'allowed_elements': os.environ.get('GOAL_ALLOWED_ELEMENTS', 'C,H,N,O,F').split(','),
    },
    'guidelines': ['valid SMILES', 'synthesizable', 'low cost'],
}


def goal_text(goal: dict) -> str:
    c = goal['constraints']
    return (f"GOAL of the team: {goal['objective']}.\n"
            f"Hard constraints: at most {c['max_heavy_atoms']} heavy (non-hydrogen) atoms; "
            f"only the elements {', '.join(c['allowed_elements'])}.\n"
            f"Also consider: {', '.join(goal['guidelines'])}.")


def start_distiller(pool: str):
    """[falda] FALDA's own distillation sidecar (T0 -> T1 -> T2 -> T3), one per campaign.
    Its cadence is scaled to a campaign of tens of minutes (defaults: 1 h and 6 h)."""
    import subprocess
    import sys
    home = f'/tmp/falda-distiller-{pool}'
    os.makedirs(os.path.join(home, '.falda'), exist_ok=True)
    env = dict(os.environ)
    env.update({
        'HOME': home,
        'FALDA_URL': os.environ.get('FALDA_URL', 'http://falda:8077'),
        'FALDA_TENANT': 'distiller', 'FALDA_POOL': pool,
        'LLM_BASE_URL': os.environ.get('FALDA_DISTILLER_BASE_URL',
                                       os.environ.get('OLLAMA_URL', 'http://localhost:11434') + '/v1'),
        'LLM_API_KEY': os.environ.get('FALDA_DISTILLER_API_KEY', 'ollama'),
        'DISTILLER_MODEL': os.environ.get('FALDA_DISTILLER_MODEL', os.environ.get('OLLAMA_MODEL', '')),
        'L1_EVERY_N': os.environ.get('FALDA_L1_EVERY_N', '6'),
        'L2_INTERVAL_S': os.environ.get('FALDA_L2_INTERVAL_S', '300'),
        'L3_INTERVAL_S': os.environ.get('FALDA_L3_INTERVAL_S', '900'),
        'DISTILLER_POLL_S': os.environ.get('FALDA_POLL_S', '30'),
    })
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'falda_memory', 'falda_distiller.py')
    return subprocess.Popen([sys.executable, script], env=env)


def falda_stats(url: str, pool: str) -> dict:
    """[falda] Size of each tier of the campaign pool at the end."""
    m = FaldaMemory(url, 'orchestrator', pool)
    try:
        return {'stream': m.call('/stream/query', {'limit': 1}).get('total'),
                'atoms': m.call('/atoms/query', {'limit': 1}).get('total'),
                'scenes': m.call('/scenes/ls', {'prefix': ''}).get('total'),
                'core_chars': len(m.call('/core/read', {}).get('content') or '')}
    except Exception as exc:
        return {'error': str(exc)}


async def main() -> int:
    init_logging(logging.INFO)
    seeds = os.environ.get('SEEDS', 'CNC(N)=O,CC1=C(O)N=C(O)N1').split(',')  # [gaied-docker patch]
    mode = os.environ.get('MEMORY_MODE', 'none').strip().lower()  # [falda] none | falda

    # [falda] measurement recorder (Goal, run metadata), then the FALDA campaign pool
    recorder = Recorder.from_env('orchestrator')
    if recorder is not None:
        recorder.set_goal({**GOAL, 'seeds': seeds})
        recorder.set_meta(mode=mode, seeds=seeds, condition=os.environ.get('EXPERIMENT_CONDITION', mode),
                          max_iterations=int(os.environ.get('MAX_ITERATIONS', '20')),
                          llm=os.environ.get('LLM_MODEL') or os.environ.get('OLLAMA_MODEL'))
    pool, distiller = None, None
    falda_url = os.environ.get('FALDA_URL', 'http://falda:8077')
    if mode == 'falda':
        pool = FaldaMemory.declare_campaign_pool(falda_url, goal_text(GOAL))
        print(f'FALDA memory: pool {pool} at {falda_url}')
        if os.environ.get('FALDA_DISTILL', 'true').lower() == 'true':
            distiller = start_distiller(pool)
    else:
        print('No shared memory (MEMORY_MODE=none)')

    config = Config(
        executors=[
            HighThroughputExecutor(
                provider=LocalProvider(
                    worker_init=(
                        ''  # [gaied-docker patch] no conda activate needed in Docker
                    ),
                ),
                # [gaied-fix] one Parsl worker per agent (was 2: a third agent never started)
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
        agents = []
        for molecule in seeds:
            agents.append(
                await manager.launch(
                    XTBSimulationAgent,
                    args=(
                        XTBConfig(),
                        molecule,
                        int(os.environ.get('MAX_ITERATIONS', '20')),   # max_iterations  # [gaied-docker patch]
                        int(os.environ.get('NO_IMPROVEMENT_TOLERANCE', '3')) or None,   # [gaied-fix] 0 = off
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

    if distiller is not None:  # [falda]
        distiller.terminate()
    if recorder is not None:
        if pool:
            stats = falda_stats(falda_url, pool)
            recorder.set_meta(falda=stats)
            print(f'FALDA pool {pool}: {stats}')
        export_path = os.environ.get('BENCH_EXPORT')
        if export_path:
            import json
            os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
            with open(export_path, 'w') as fh:
                json.dump(recorder.export(), fh)
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