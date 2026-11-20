"""Offline reset bank via Ray Data + load-time benchmark.

1. Ray Data cooks all released OmniReset reset datasets (6 object pairs x 4 reset types) into sharded parquet
   under BANK_DIR (pair=<pair>/shard=<0..23>/). The pipeline orchestration runs in a CPU task on a GPU node
   (the head has no numpy/pyarrow and 0 CPUs); every Ray Data operator carries a label_selector that keeps it
   off broken nodes and off the node given in --avoid.
2. Benchmark on 1 GPU: OmniReset's original load path for the task's pair (torch.load of the 4 .pt files +
   the same per-element stack MultiResetManager does) vs reading cooked shards (full bank, and 1 worker's
   share at W=6). Equivalence check: cooked values == original values for every state.
"""
import argparse
import os
import shutil
import time

import ray

import config
from common import RunMeta, write_json

TASK_PAIR = "Peg__PegHole"


@ray.remote(num_cpus=0)
def orchestrate(out_dir, rra):
    import bank_pipeline
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    return bank_pipeline.run_pipeline(out_dir, ray_remote_args=rra)


@ray.remote(num_gpus=1, num_cpus=4)
def benchmark(bank_dir, pair, num_workers):
    import os
    import socket
    import urllib.request

    import torch

    import bank_pipeline as bp
    res = dict(host=socket.gethostname())

    # (a) OmniReset's original path: torch.load each file + MultiResetManager's sample_state_data_set stacking
    def sample_state_data_set(d, idx, device):  # verbatim logic of uwlab events.sample_state_data_set
        out = {}
        for k, v in d.items():
            out[k] = sample_state_data_set(v, idx, device) if isinstance(v, dict) else \
                torch.stack([v[i] for i in idx.tolist()], dim=0).to(device)
        return out

    t_dl = t_load = t_stack = 0.0
    orig = {}
    for rt in bp.RESET_TYPES:
        path = f"/tmp/omnireset_{pair}_{rt}.pt"
        t = time.time()
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{bp.HF_BASE}/{pair}/resets_{rt}.pt", path)
        t_dl += time.time() - t
        t = time.time()
        d = torch.load(path, weights_only=False)
        t_load += time.time() - t
        t = time.time()
        n = len(d["initial_state"]["articulation"]["robot"]["joint_position"])
        orig[rt] = sample_state_data_set(d["initial_state"], torch.arange(n), "cuda:0")
        torch.cuda.synchronize()
        t_stack += time.time() - t
    res["original"] = dict(download_s=t_dl, torch_load_s=t_load, stack_to_gpu_s=t_stack,
                           total_excl_download_s=t_load + t_stack,
                           rows=sum(len(v["articulation"]["robot"]["joint_position"]) for v in orig.values()))

    # (b) cooked shards: full bank for the pair (W=1) and one worker's share at W=num_workers
    st, ids, tid, tm_all = bp.load_worker_shard(bank_dir, pair, 0, 1)
    res["cooked_full"] = tm_all
    _, _, _, tm_w = bp.load_worker_shard(bank_dir, pair, 0, num_workers)
    res[f"cooked_worker0_of_{num_workers}"] = tm_w

    # (c) equivalence: every cooked state equals the original state with the same (type, index)
    max_err, checked = 0.0, 0
    local = (ids % 1_000_000).long()
    for ti, rt in enumerate(bp.RESET_TYPES):
        m = tid == ti
        for a, b, f, _ in bp.LEAVES:
            ref = orig[rt][a][b][f][local[m]]
            got = st[a][b][f][m]
            max_err = max(max_err, float((ref - got).abs().max()))
        checked += int(m.sum())
    res["equivalence"] = dict(states_checked=checked, max_abs_err=max_err, exact=max_err == 0.0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avoid", nargs="*", default=[], help="node-id prefixes to keep this run off")
    args = ap.parse_args()
    ray.init(address=config.RAY_ADDRESS)
    meta = RunMeta("phase3")
    # exclude broken nodes, the --avoid nodes, and the head (no numpy/pandas; num_cpus=0 tasks can land there)
    bad = [n["NodeID"] for n in ray.nodes()
           if any(n["NodeID"].startswith(p) for p in config.EXCLUDE_NODE_ID_PREFIXES + args.avoid)
           or not n["Resources"].get("GPU")]
    sel = {"ray.io/node-id": f"!in({','.join(bad)})"} if bad else None
    rra = {"label_selector": sel} if sel else {}
    usable = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU") and n["NodeID"] not in bad]

    print(f"Ray Data reset-bank pipeline -> {config.BANK_DIR} on up to {len(usable)} nodes")
    ref = orchestrate.options(label_selector=sel).remote(config.BANK_DIR, rra)
    ready, _ = ray.wait([ref], timeout=3600)
    if not ready:
        raise SystemExit("pipeline timed out")
    st = ray.get(ready[0])
    print(f"rows in {st['rows_in']:,} -> valid {st['rows_out']:,} ({st['rows_invalid']} invalid: {st['invalid_by_reason']})")
    print(f"files {st['files']} | {st['columns']} float32 state columns | {st['num_shards']} shards per pair")
    print(f"pipeline wall {st['pipeline_wall_s']:.1f}s (cook {st['cook_wall_s']:.1f}s, write {st['write_wall_s']:.1f}s) "
          f"-> {st['rows_per_s']:,.0f} rows/s")
    sc = st["stage_cpu_s"]
    print(f"stage task-seconds summed over parallel tasks: download {sc['download']:.1f} | torch.load "
          f"{sc['torch_load']:.1f} | flatten {sc['flatten']:.1f}")
    print(f"by stratum: {st['rows_by_stratum']}")

    W = config.NUM_WORKERS
    b = ray.get(benchmark.options(label_selector=sel).remote(config.BANK_DIR, TASK_PAIR, W), timeout=1800)
    o, cf, cw = b["original"], b["cooked_full"], b[f"cooked_worker0_of_{W}"]
    print(f"\nload benchmark on {b['host'][:22]} ({TASK_PAIR}, {o['rows']:,} states):")
    print(f"  original  torch.load {o['torch_load_s']:.2f}s + per-element stack->GPU {o['stack_to_gpu_s']:.2f}s "
          f"= {o['total_excl_download_s']:.2f}s")
    print(f"  cooked, full bank     {cf['total_s']:.3f}s ({cf['rows']:,} rows, {cf['files']} files) "
          f"-> {o['total_excl_download_s']/cf['total_s']:.0f}x faster")
    print(f"  cooked, worker 0 of {W} {cw['total_s']:.3f}s ({cw['rows']:,} rows)")
    print(f"  equivalence: {b['equivalence']}")
    print("  (Phase 1 measured the original torch.load *inside the Isaac actor* at 69.5 s cold / 68.3 s warm.)")

    out = dict(meta=meta.finish(gpu_nodes_used=len(usable)), pipeline=st, benchmark=b, task_pair=TASK_PAIR,
               note="Bank = OmniReset's released reset states (all 6 pairs, 4 types each), cooked, validated, "
                    "sharded. Workers for the peg task read pair=Peg__PegHole only. No synthetic states added.")
    m = out["meta"]
    print(f"\nwall {m['wall_s']:.0f}s | est. ${m['usd_used_nodes']:.2f} ({len(usable)} nodes + head) | "
          f"${m['usd_whole_cluster']:.2f} whole cluster")
    write_json("bank_stats.json", out)


if __name__ == "__main__":
    main()
