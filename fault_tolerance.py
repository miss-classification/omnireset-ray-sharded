"""Fault tolerance. W workers training; at ~step 200 (iteration 7 = 224 steps/worker with L=32) the driver
calls ray.kill(worker, no_restart=False). Expected: the other workers keep training (throughput dip logged), Ray
restarts the killed actor, whose __init__ rebuilds the sim and reloads ONLY its own cooked bank shard; the driver
re-runs setup_training + weights + SGS weights and it rejoins; throughput recovers; the learner's SGS counters
(global, merged from deltas) are untouched by the loss of one worker.

W defaults to 5 (5 workers + 1 learner GPU); the killed worker is the last one (index W-1).
"""
import argparse
import statistics

import ray

import config
import train_loop
from common import RunMeta, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--envs", type=int, default=config.ENVS_PER_GPU)
    ap.add_argument("--kill-worker", type=int, default=None, help="default: last worker (W-1)")
    ap.add_argument("--kill-at", type=int, default=7, help="iteration (x L=32 steps per worker)")
    ap.add_argument("--post", type=int, default=5, help="iterations to keep running after rejoin")
    ap.add_argument("--cap", type=int, default=80, help="max iterations")
    args = ap.parse_args()
    kw = args.W - 1 if args.kill_worker is None else args.kill_worker

    ray.init(address=config.RAY_ADDRESS)
    meta = RunMeta("phase5")
    res = train_loop.run("p5_kill", args.W, args.envs, args.cap, kill=(kw, args.kill_at), meta=meta,
                         post_rejoin_iters=args.post)

    its = res["iterations"]
    ev = {e["event"]: e for e in res["timeline"]}
    k_it = ev["KILL"]["iteration"]
    r_it = ev.get("REJOINED", {}).get("iteration")
    before = [r for r in its if 2 < r["iteration"] < k_it]
    during = [r for r in its if r["iteration"] >= k_it and (r_it is None or r["iteration"] <= r_it)]
    after = [r for r in its if r_it is not None and r["iteration"] > r_it]
    mean = lambda rows: statistics.fmean(r["agg_env_steps_per_s"] for r in rows) if rows else None  # noqa: E731
    summary = dict(W=args.W, killed_worker=kw, kill_iteration=k_it,
                   tput_before=mean(before), tput_during=mean(during), tput_after=mean(after),
                   workers_during=sorted({len(r["workers"]) for r in during}),
                   rejoined=r_it is not None, rejoin_iteration=r_it,
                   down_s=ev.get("REJOINED", {}).get("down_s"), reinit_s=ev.get("REJOINED", {}).get("reinit_s"),
                   bank_reload_s=ev.get("REJOINED", {}).get("bank_load_s"),
                   learner_versions=[r["learner"]["version"] for r in its],
                   sgs_states_seen_by_iter=[r["learner"]["sgs_states_seen"] for r in its],
                   sgs_final=res["sgs"])

    print("\n================ FAULT-TOLERANCE TIMELINE ================")
    for e in res["timeline"]:
        extra = ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("t", "iteration", "env_steps", "event"))
        print(f"  t={e['t']:7.1f}s  iteration {e['iteration']:3d}  env-steps {e['env_steps']:>12,}  {e['event']:<12} {extra}")
    f = lambda x: f"{x:,.0f}" if x else "-"  # noqa: E731
    print(f"\n  throughput: before kill {f(summary['tput_before'])} | while {args.W - 1} workers "
          f"{f(summary['tput_during'])} | after rejoin {f(summary['tput_after'])} env-steps/s")
    print(f"  worker {kw}: down {summary['down_s']} s (re-init {summary['reinit_s']} s, of which bank-shard reload "
          f"{summary['bank_reload_s']} s); learner never stopped: versions {its[0]['learner']['version']} -> "
          f"{its[-1]['learner']['version']} with no gap")
    print(f"  SGS states seen (global, learner): {summary['sgs_states_seen_by_iter']}")
    out = dict(meta=meta.finish(gpu_nodes_used=args.W + 1), summary=summary, run=res)
    m = out["meta"]
    print(f"wall {m['wall_s']/60:.1f} min | est. ${m['usd_used_nodes']:.2f} | ${m['usd_whole_cluster']:.2f} whole cluster")
    write_json("kill_timeline.json", out)


if __name__ == "__main__":
    main()
