"""Teardown: kill every demo actor, then check every reachable node for leftover Isaac/Kit processes.

Run after each phase (or any time). Prints what is still running. Exit code 1 if anything survived.
"""
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_actors

import config
from sim_worker import orphan_check

DEMO_CLASSES = ("SimWorker", "Learner")


def main():
    ray.init(address=config.RAY_ADDRESS, logging_level="ERROR")

    # 1. kill demo actors (named or not) still alive in any job
    alive = [a for a in list_actors(filters=[("state", "=", "ALIVE")], limit=10000)
             if a.class_name in DEMO_CLASSES]
    print(f"alive demo actors: {len(alive)}")
    for a in alive:
        try:
            h = ray.get_actor(a.name, namespace=a.ray_namespace) if a.name else None
        except ValueError:
            h = None
        if h is not None:
            ray.kill(h, no_restart=True)
            print(f"  killed {a.class_name} name={a.name} pid={a.pid} node={a.node_id[:8]}")
        else:
            print(f"  cannot get a handle for unnamed {a.class_name} pid={a.pid} node={a.node_id[:8]} "
                  f"(owner job {a.job_id}); it dies when its driver exits")
    if alive:
        time.sleep(5)

    # 2. cluster-wide orphan check on every alive node we can schedule on
    nodes = [n for n in ray.nodes() if n["Alive"]]
    excluded = [n for n in nodes if any(n["NodeID"].startswith(p) for p in config.EXCLUDE_NODE_ID_PREFIXES)]
    reachable = [n for n in nodes if n not in excluded]
    refs = {orphan_check.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))
            .remote(): n for n in reachable}
    ready, pending = ray.wait(list(refs), num_returns=len(refs), timeout=120)
    survivors = 0
    for r in ready:
        host, procs = ray.get(r)
        survivors += len(procs)
        print(f"{host[:24]:<24} isaac/kit procs: {len(procs)}")
        for p in procs:
            print("    ", p[:170])
    for r in pending:
        print(f"{refs[r]['NodeManagerHostname'][:24]:<24} no answer within 120s")
    for n in excluded:
        print(f"{n['NodeManagerHostname'][:24]:<24} excluded (broken node, cannot run checks)")

    # 3. what Ray still has allocated
    total, avail = ray.cluster_resources(), ray.available_resources()
    print(f"GPU in use: {total.get('GPU', 0) - avail.get('GPU', 0):.1f} / {total.get('GPU', 0):.0f} | "
          f"CPU in use: {total.get('CPU', 0) - avail.get('CPU', 0):.1f} / {total.get('CPU', 0):.0f}")
    still = [a for a in list_actors(filters=[("state", "=", "ALIVE")], limit=10000) if a.class_name in DEMO_CLASSES]
    print(f"demo actors still alive: {len(still)} | orphan Isaac/Kit processes: {survivors}")
    sys.exit(1 if (still or survivors) else 0)


if __name__ == "__main__":
    main()
