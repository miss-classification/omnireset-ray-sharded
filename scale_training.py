"""Multi-worker scaling with the cooked reset bank, per-state SGS and the PPO learner in the loop.

Runs train_loop.run() for each W in --ws (default 1 2 4 5; 6 healthy GPUs = W workers + 1 learner GPU), each worker
at config.ENVS_PER_GPU envs, >= 300 env steps per worker (ITER iterations x L=32). Per W: init wall (parallel) vs
sum of per-worker init (what a serialized startup would cost), steady-state aggregate env-steps/s (iterations >= 3),
efficiency vs linear from W=1, learner update / ingest time, SGS merge time + delta bytes, rollout buffer MB.

Also a matched-total-envs startup comparison: 1 worker x N vs 4 workers x N/4 (same total envs).

--smoke: W=2 x 1024 envs x 3 iterations, to validate the loop before the expensive runs.
"""
import argparse
import statistics

import ray

import config
import train_loop
from common import RunMeta, write_json

WARMUP_ITERS = 2


def summarize(res):
    it = [r for r in res["iterations"] if r["iteration"] > WARMUP_ITERS] or res["iterations"]
    tput = statistics.fmean(r["agg_env_steps_per_s"] for r in it)
    return dict(
        W=res["W"], envs_per_worker=res["envs_per_worker"], total_envs=res["total_envs"],
        agg_env_steps_per_s=tput,
        per_worker_env_steps_per_s=statistics.fmean(
            statistics.fmean(r["per_worker_env_steps_per_s"].values()) for r in it),
        init_wall_s=res["init_wall_s"], init_max_s=max(res["init_each_s"].values()),
        init_sum_s=sum(res["init_each_s"].values()),
        bank_load_s_max=max(t.get("bank_load_s", 0) for t in res["timings_each"].values()),
        learner_update_s=statistics.fmean(r["learner"]["update_s"] for r in it),
        learner_ingest_s=statistics.fmean(r["learner"]["ingest_s"] for r in it),
        sgs_merge_ms=1e3 * statistics.fmean(r["learner"]["sgs_merge_s"] for r in it),
        sgs_delta_kb=statistics.fmean(r["learner"]["sgs_delta_bytes"] for r in it) / 1024,
        buffer_mb_per_iter=statistics.fmean(r["buffer_mb"] for r in it),
        iter_wall_s=statistics.fmean(r["wall_s"] for r in it),
        steady_iters=len(it), sgs=res["sgs"],
        success_rate_last=(res["iterations"][-1]["success"] / res["iterations"][-1]["attempts"]
                           if res["iterations"][-1]["attempts"] else None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", type=int, nargs="*", default=[1, 2, 4, 5])
    ap.add_argument("--envs", type=int, default=config.ENVS_PER_GPU)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--matched", action="store_true", help="also run the matched-total-envs startup comparison")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.ws, args.envs, args.iters, args.matched = [2], 1024, 3, False

    ray.init(address=config.RAY_ADDRESS)
    meta = RunMeta("phase4_smoke" if args.smoke else "phase4")
    runs, rows = [], []
    for W in args.ws:
        res = train_loop.run(f"p4{'smoke' if args.smoke else ''}_W{W}", W, args.envs, args.iters, meta=meta)
        runs.append(res)
        rows.append(summarize(res))
    if args.matched:  # 1 x N is the W=1 row above; add 4 x N/4 (same total envs)
        n = args.envs
        for W, e in ((4, n // 4),):
            res = train_loop.run(f"p4matched_W{W}x{e}", W, e, 2, meta=meta)
            res["matched_startup"] = True
            runs.append(res)
            rows.append(dict(summarize(res), matched_startup=True))

    base = next((r for r in rows if r["W"] == 1 and not r.get("matched_startup")), None)
    print(f"\n{'W':>2} {'total_envs':>10} {'agg env-steps/s':>16} {'eff vs linear':>13} {'init wall':>9} "
          f"{'init sum':>9} {'bank load':>9} {'update s':>8} {'SGS ms':>7} {'buf MB':>7}")
    for r in rows:
        eff = r["agg_env_steps_per_s"] / (r["W"] * base["agg_env_steps_per_s"]) if base and not r.get("matched_startup") else None
        r["efficiency_vs_linear"] = eff
        print(f"{r['W']:>2} {r['total_envs']:>10,} {r['agg_env_steps_per_s']:>16,.0f} "
              f"{(f'{eff:.2f}' if eff else '-'):>13} {r['init_wall_s']:>9.0f} {r['init_sum_s']:>9.0f} "
              f"{r['bank_load_s_max']:>9.2f} {r['learner_update_s']:>8.2f} {r['sgs_merge_ms']:>7.1f} "
              f"{r['buffer_mb_per_iter']:>7.0f}{'  (matched-envs startup run)' if r.get('matched_startup') else ''}")
    gpus = max(r["W"] for r in rows) + 1
    out = dict(meta=meta.finish(gpu_nodes_used=gpus), envs_per_worker=args.envs, rollout_len=config.ROLLOUT_LEN,
               rows=rows, runs=runs, paper_envs_per_gpu=config.PAPER_ENVS_PER_GPU)
    m = out["meta"]
    print(f"wall {m['wall_s']/60:.1f} min | est. ${m['usd_used_nodes']:.2f} ({gpus} GPU nodes + head) | "
          f"${m['usd_whole_cluster']:.2f} whole cluster")
    write_json("scaling_smoke.json" if args.smoke else "scaling.json", out)


if __name__ == "__main__":
    main()
