"""Single GPU: two ways to cut startup on one GPU, run concurrently on two nodes.

mps      k simulators x N/k envs share one GPU with NVIDIA MPS on (concurrent kernels across processes). Phase 10
         without MPS: startup 2.5x faster but stepping 0.45x (time-sliced contexts). Does MPS recover stepping?
overlap  simulator A (N envs) steps continuously; simulator B (N envs) starts on the SAME GPU meanwhile (startup is
         CPU work: GPU ~0.4% busy during init). B's startup vs standalone, A's throughput before/during B's startup.

MPS: a control daemon is started on the node by a Ray task (as the ray user, pipe dir in /tmp), the actors get the
same CUDA_MPS_PIPE_DIRECTORY, and the daemon is stopped afterwards. Output: one_gpu.json
"""
import argparse
import concurrent.futures as cf
import statistics
import time

import ray

import config
from common import RunMeta, write_json
from sim_worker import SimWorker

MPS_ENV = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps-pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-mps-log"}


@ray.remote(num_cpus=0)
def mps(cmd):
    import os
    import subprocess
    env = dict(os.environ, **MPS_ENV)
    for d in MPS_ENV.values():
        os.makedirs(d, exist_ok=True)
    if cmd == "start":
        r = subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, capture_output=True, text=True)
    else:
        r = subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", env=env, capture_output=True, text=True)
    time.sleep(2)
    ps = subprocess.run("pgrep -a nvidia-cuda-mps", shell=True, capture_output=True, text=True).stdout
    return dict(rc=r.returncode, out=(r.stdout + r.stderr)[-300:], procs=ps.strip())


def worker(name, wid, n, k, node, frac, extra_env=None):
    env = dict(config.ISAAC_ENV_VARS, **(extra_env or {}))
    return SimWorker.options(name=name, num_gpus=frac, max_restarts=0, label_selector={"ray.io/node-id": node},
                             runtime_env={"env_vars": env}) \
        .remote(wid, n, config.TASK, config.TASK_OVERRIDES, config.worker_seed(wid), config.AGENT_ENTRY,
                config.BANK_DIR, config.TELEMETRY_INTERVAL_S, config.BANK_PAIR, k)


def close(ws):
    for w in ws:
        try:
            ray.wait([w.close.remote()], timeout=120)
        except Exception:  # noqa: BLE001
            pass
        ray.kill(w, no_restart=True)


def run_mps(node, total, k):
    rec = dict(name=f"mps_{k}x{total // k}", node=node, ok=False)
    rec["mps_start"] = ray.get(mps.options(label_selector={"ray.io/node-id": node}).remote("start"), timeout=120)
    t0 = time.time()
    ws = [worker(f"p11mps_{i}", i, total // k, k, node, 1.0 / k, MPS_ENV) for i in range(k)]
    try:
        ray.get([w.ping.remote() for w in ws], timeout=3600)
        rec["init_wall_s"] = time.time() - t0
        ray.get([w.run_random.remote(10) for w in ws], timeout=1800)
        t = time.time()
        steps = ray.get([w.run_random.remote(60) for w in ws], timeout=3600)
        wall = time.time() - t
        st = ray.get([w.get_stats.remote() for w in ws], timeout=300)
        rec.update(ok=True, agg_env_steps_per_s=total * 60 / wall,
                   per_process=[s["env_steps_per_s"] for s in steps], gpu_used_mib=max(s["gpu_used_mib"] for s in st),
                   physics_start_s=[s["timings"].get("stage_sim_reset_s") for s in st],
                   sm_pct_step=[(s["telemetry"].get("step") or {}).get("sm_pct", {}).get("mean") for s in st])
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[-1000:]}"
    finally:
        close(ws)
        rec["mps_stop"] = ray.get(mps.options(label_selector={"ray.io/node-id": node}).remote("stop"), timeout=120)
    return rec


def run_overlap(node, n, chunk=20):
    rec = dict(name=f"overlap_{n}", node=node, ok=False, a_timeline=[])
    a = worker("p11_A", 0, n, 1, node, 0.5)
    b = None
    try:
        t0 = time.time()
        ray.get(a.ping.remote(), timeout=3600)
        rec["a_init_s"] = time.time() - t0
        ray.get(a.run_random.remote(10), timeout=1800)

        def step_a(phase):
            r = ray.get(a.run_random.remote(chunk), timeout=1800)
            rec["a_timeline"].append(dict(t=time.time() - t0, phase=phase, env_steps_per_s=r["env_steps_per_s"]))

        for _ in range(3):
            step_a("alone_before")
        tb = time.time()
        b = worker("p11_B", 0, n, 1, node, 0.5)  # worker_id 0 of 1: owns all bank shards
        pb = b.ping.remote()
        while True:
            step_a("during_B_startup")
            if ray.wait([pb], timeout=0)[0]:
                break
            if time.time() - tb > 3600:
                raise TimeoutError("B startup > 3600 s")
        ray.get(pb)
        rec["b_startup_while_a_trains_s"] = time.time() - tb
        rec["b_timings"] = ray.get(b.get_stats.remote(), timeout=300)["timings"]
        close([b])
        b = None
        for _ in range(3):
            step_a("alone_after")
        m = lambda p: statistics.fmean(x["env_steps_per_s"] for x in rec["a_timeline"] if x["phase"] == p)  # noqa: E731
        rec.update(ok=True, a_tput_alone=statistics.fmean([m("alone_before"), m("alone_after")]),
                   a_tput_during_b_startup=m("during_B_startup"))
        rec["a_slowdown_during_b_startup"] = 1 - rec["a_tput_during_b_startup"] / rec["a_tput_alone"]
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[-1000:]}"
    finally:
        close([w for w in (a, b) if w is not None])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16384)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--only", choices=["mps", "overlap"], default=None)
    args = ap.parse_args()
    ray.init(address=config.RAY_ADDRESS)
    meta = RunMeta("phase11")
    bad = set(n["NodeID"] for n in ray.nodes() if any(n["NodeID"].startswith(p) for p in config.EXCLUDE_NODE_ID_PREFIXES))
    nodes = [n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) >= 1 and n["NodeID"] not in bad]
    jobs = {}
    if args.only in (None, "mps"):
        jobs["mps"] = lambda: run_mps(nodes[2], args.envs, args.k)
    if args.only in (None, "overlap"):
        jobs["overlap"] = lambda: run_overlap(nodes[3], args.envs)
    with cf.ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        res = {k: f.result() for k, f in futs.items()}
    if "mps" in res:
        r = res["mps"]
        print(f"\nMPS {r['name']}: ok={r['ok']} startup {r.get('init_wall_s', 0):.0f}s  agg {r.get('agg_env_steps_per_s', 0):,.0f} "
              f"env-steps/s  VRAM {r.get('gpu_used_mib', 0) / 1024:.1f} GiB  physics start {r.get('physics_start_s')}  "
              f"SM% {r.get('sm_pct_step')}  daemon {r['mps_start'].get('procs')!r} {r.get('error', '')[:300]}")
        print("   (Phase 10, no MPS, same split: startup 210 s, 8,347 env-steps/s; 1 x 16,384: 532 s, 18,719)")
    if "overlap" in res:
        r = res["overlap"]
        print(f"\nOverlap: ok={r['ok']} B startup while A trains {r.get('b_startup_while_a_trains_s', 0):.0f}s "
              f"(standalone 1 x 16,384 ~532 s) | A alone {r.get('a_tput_alone', 0):,.0f} env-steps/s, during B startup "
              f"{r.get('a_tput_during_b_startup', 0):,.0f} ({100 * r.get('a_slowdown_during_b_startup', 0):.0f}% slower) "
              f"{r.get('error', '')[:300]}")
    out = dict(meta=meta.finish(gpu_nodes_used=len(jobs)), results=res)
    m = out["meta"]
    print(f"wall {m['wall_s']/60:.1f} min | est. ${m['usd_used_nodes']:.2f} | ${m['usd_whole_cluster']:.2f} whole cluster")
    write_json("one_gpu.json", out)


if __name__ == "__main__":
    main()
