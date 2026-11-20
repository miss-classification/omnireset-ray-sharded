"""GPU placement check.

Launches NUM_GPUS_EXPECTED concurrent @ray.remote(num_gpus=1) tasks. Each reports
hostname, Ray GPU ids, CUDA_VISIBLE_DEVICES, nvidia-smi, and probes the node's
software environment (without importing isaacsim/isaaclab). Asserts:
  - every task got exactly 1 GPU
  - no task ran on the head node
  - no (node, gpu) pair is shared  -> 1 task per GPU
  - all tasks overlapped in time  -> genuinely concurrent
Writes placement.json.
"""
import json
import os
import socket
import time

import ray

import config

HOLD_S = 10  # each task holds its GPU this long so all 8 must coexist


@ray.remote(num_gpus=1)
def probe(i: int):
    import importlib.util
    import subprocess
    import sys

    t0 = time.time()

    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"ERR {e}"

    def has(mod):
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:  # noqa: BLE001
            return False

    pkgs = sh("pip list 2>/dev/null | grep -iE '^(isaacsim|isaaclab|torch|rsl-rl-lib|rsl_rl|ray) '")
    out = dict(
        idx=i,
        hostname=socket.gethostname(),
        node_id=ray.get_runtime_context().get_node_id(),
        ray_gpu_ids=[str(g) for g in ray.get_gpu_ids()],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        nvidia_smi=sh("nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader"),
        python=sys.version.split()[0],
        pip=pkgs,
        has_isaacsim=has("isaacsim"),
        has_isaaclab=has("isaaclab"),
        has_torch=has("torch"),
        isaaclab_dir_exists=os.path.isdir(config.ISAACLAB_DIR),
        omni_kit_cachedir=os.environ.get("OMNI_KIT_CACHEDIR"),
        eula_vars={k: v for k, v in os.environ.items() if "EULA" in k.upper()},
        cluster_storage_ok=os.path.isdir(config.STORAGE),
    )
    time.sleep(HOLD_S)
    out["t_start"], out["t_end"] = t0, time.time()
    return out


def head_node_ids():
    ids = set()
    for n in ray.nodes():
        if not n["Alive"]:
            continue
        res = n.get("Resources", {})
        if n.get("IsHeadNode") or "anyscale/node-group:head" in res or res.get("GPU", 0) == 0:
            ids.add(n["NodeID"])
    return ids


def main():
    ray.init(address=config.RAY_ADDRESS)
    n = config.NUM_GPUS_EXPECTED
    total_gpus = ray.cluster_resources().get("GPU", 0)
    print(f"cluster GPUs: {total_gpus}  launching {n} probes")
    heads = head_node_ids()

    t0 = time.time()
    refs = [probe.remote(i) for i in range(n)]
    ready, pending = ray.wait(refs, num_returns=n, timeout=900)
    if pending:
        raise SystemExit(f"FAIL: {len(pending)} probes did not finish in 900s (not enough GPUs schedulable?)")
    res = sorted(ray.get(ready, timeout=60), key=lambda r: r["idx"])
    wall = time.time() - t0

    print(f"\n{'#':>2} {'hostname':<44} {'ray_gpu':<7} {'CVD':<4} {'nvidia-smi'}")
    for r in res:
        print(f"{r['idx']:>2} {r['hostname']:<44} {','.join(r['ray_gpu_ids']):<7} "
              f"{str(r['cuda_visible_devices']):<4} {r['nvidia_smi']}")

    # ---- assertions ----
    errs = []
    for r in res:
        if len(r["ray_gpu_ids"]) != 1:
            errs.append(f"probe {r['idx']} has gpu ids {r['ray_gpu_ids']}")
        if r["node_id"] in heads:
            errs.append(f"probe {r['idx']} ran on head node {r['hostname']}")
        if not r["nvidia_smi"] or r["nvidia_smi"].startswith("ERR"):
            errs.append(f"probe {r['idx']} has no nvidia-smi output")
    pairs = [(r["node_id"], r["ray_gpu_ids"][0] if r["ray_gpu_ids"] else None) for r in res]
    if len(set(pairs)) != len(pairs):
        errs.append("two probes share a (node, gpu) pair")
    overlap = min(r["t_end"] for r in res) - max(r["t_start"] for r in res)
    if overlap <= 0:
        errs.append(f"probes were not concurrent (overlap {overlap:.1f}s)")

    nodes = sorted({r["hostname"] for r in res})
    print(f"\nwall {wall:.1f}s | concurrent overlap {overlap:.1f}s | {len(nodes)} distinct GPU node(s) | head nodes excluded: {len(heads)}")

    # ---- environment on GPU nodes (informational, gates Phase 1) ----
    env_keys = ["python", "pip", "has_isaacsim", "has_isaaclab", "has_torch", "isaaclab_dir_exists",
                "omni_kit_cachedir", "eula_vars", "cluster_storage_ok"]
    envs = {json.dumps({k: r[k] for k in env_keys}, sort_keys=True) for r in res}
    print(f"\nGPU-node software environment ({len(envs)} distinct variant(s)):")
    for e in envs:
        for k, v in json.loads(e).items():
            print(f"  {k}: {v}")

    out = dict(ok=not errs, errors=errs, wall_s=wall, overlap_s=overlap, distinct_nodes=nodes,
               cluster_gpus=total_gpus, probes=res)
    path = os.path.join(config.RESULTS_DIR, "placement.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    if errs:
        print("\nPLACEMENT FAIL:\n  " + "\n  ".join(errs))
        raise SystemExit(1)
    print(f"PLACEMENT OK: {n} tasks, 1 GPU each, none on head, all concurrent")


if __name__ == "__main__":
    main()
