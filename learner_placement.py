"""Learner placement. Own GPU vs sharing a GPU with worker 0 (num_gpus=0.5 each).

Configs (6 healthy GPUs):
  own_W5     5 workers x 1.0 GPU + learner 1.0 GPU (6 GPUs)          -- reused from Phase 4's W=5 row if present
  shared_W5  worker 0 at 0.5 + learner 0.5 on the same GPU, 4 workers x 1.0 (5 GPUs)  -> same W, one GPU freed
  shared_W6  worker 0 at 0.5 + learner 0.5, 5 workers x 1.0 (6 GPUs) -> the extra sim GPU the sharing buys
Only one Kit process per GPU in every config (the learner is not an Isaac process). If sharing is unstable (OOM,
crash, hang), that is recorded and the config is skipped: the negative result is useful.
"""
import argparse
import json
import os
import statistics

import ray

import config
import train_loop
from common import RunMeta, write_json


def tput(res):
    its = [r for r in res["iterations"] if r["iteration"] > 2] or res["iterations"]
    return dict(agg_env_steps_per_s=statistics.fmean(r["agg_env_steps_per_s"] for r in its),
                worker0_env_steps_per_s=statistics.fmean((r["per_worker_env_steps_per_s"].get(0) or r["per_worker_env_steps_per_s"].get("0", 0)) for r in its),
                others_env_steps_per_s=statistics.fmean(
                    statistics.fmean(v for k, v in r["per_worker_env_steps_per_s"].items() if str(k) != "0") for r in its),
                learner_update_s=statistics.fmean(r["learner"]["update_s"] for r in its),
                iter_wall_s=statistics.fmean(r["wall_s"] for r in its))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=config.ENVS_PER_GPU)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    ray.init(address=config.RAY_ADDRESS)
    meta = RunMeta("phase6")
    rows = {}
    sc = os.path.join(config.RESULTS_DIR, "scaling.json")
    if os.path.exists(sc):
        prev = json.load(open(sc))
        own = next((r for r in prev["runs"] if r["W"] == 5 and r["envs_per_worker"] == args.envs
                    and not r.get("matched_startup")), None)
        if own:
            rows["own_W5"] = dict(source="phase4 scaling.json", gpus=6, **tput(own))
    if "own_W5" not in rows:
        rows["own_W5"] = dict(gpus=6, **tput(train_loop.run("p6_own_W5", 5, args.envs, args.iters, meta=meta)))
    for name, W in (("shared_W5", 5), ("shared_W6", 6)):
        try:
            res = train_loop.run(f"p6_{name}", W, args.envs, args.iters, meta=meta, learner_gpu=0.5,
                                 worker_gpu_overrides={0: 0.5})
            same = res["placement"][0]["node_id"] == res["placement"][-1]["node_id"]
            rows[name] = dict(gpus=W if W == 6 else 5, learner_colocated_with_worker0=same, **tput(res),
                              vram_worker0_mib=res["final_worker_stats"].get("0", {}).get("gpu_used_mib"))
        except Exception as e:  # noqa: BLE001  instability is a result
            rows[name] = dict(failed=True, error=f"{type(e).__name__}: {str(e)[:500]}")
            print(f"  {name} FAILED: {rows[name]['error']}")
    print(f"\n{'config':<10} {'GPUs':>4} {'agg env-steps/s':>16} {'per GPU':>9} {'w0 env-steps/s':>15} "
          f"{'others':>9} {'update s':>8}")
    for k, r in rows.items():
        if r.get("failed"):
            print(f"{k:<10} FAILED  {r['error'][:100]}")
            continue
        print(f"{k:<10} {r['gpus']:>4} {r['agg_env_steps_per_s']:>16,.0f} {r['agg_env_steps_per_s']/r['gpus']:>9,.0f} "
              f"{r['worker0_env_steps_per_s']:>15,.0f} {r['others_env_steps_per_s']:>9,.0f} {r['learner_update_s']:>8.2f}")
    out = dict(meta=meta.finish(gpu_nodes_used=6), envs_per_worker=args.envs, rows=rows)
    m = out["meta"]
    print(f"wall {m['wall_s']/60:.1f} min | est. ${m['usd_used_nodes']:.2f} | ${m['usd_whole_cluster']:.2f} whole cluster")
    write_json("learner_placement.json", out)


if __name__ == "__main__":
    main()
