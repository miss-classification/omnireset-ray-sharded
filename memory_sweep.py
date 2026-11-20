"""Envs-per-GPU sweep for the OmniReset peg/peghole task, on 1 node.

Sweep num_envs 1024, 2048, 4096, ... (doubling) until OOM / init failure, or until per-env step time is
>2x worse than its best (throughput-degraded), whichever comes first. Each point is a fresh SimWorker
(1 Kit app per process; max_restarts=0 so a crash is a data point, not a restart loop), on the same node as
Phase 1 run A (warm caches), so points differ only in num_envs.

Per point: VRAM (whole-process peak from nvidia-smi dmon; torch allocated/reserved), init time + stage
breakdown, env-steps/s, s/step, s/step/env, SM% and mem-bw% means during stepping.

Two different ceilings, reported separately:
  * VRAM ceiling: from the linear fit fixed_overhead + per_env * N, the largest N using <= 75% of GPU memory.
  * Throughput knee: the smallest N reaching >= 90% of the best observed aggregate env-steps/s
    (beyond it, more envs add memory and latency but little throughput).
ENVS_PER_GPU (Phase 4) = min(VRAM-safe @75%, knee), rounded down to a multiple of 256.
"""
import argparse
import json
import os
import time

import ray

import config
from common import RunMeta, write_json
from sim_worker import SimWorker, orphan_check
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

GPU_TOTAL_MIB = 81920          # A100-SXM4-80GB as reported by nvidia-smi
HEADROOM = 0.75
WARMUP_STEPS = 10
MEASURE_STEPS = 60
KNEE_FRACTION = 0.90
DEGRADE_FACTOR = 2.0
# Only specific CUDA OOM strings count as VRAM OOM. A bare actor death (e.g. PhysX aborting on its 64K material
# limit, seen at 32,768 envs on 2026-11-19) is NOT a VRAM ceiling -- check the worker log before labeling it.
OOM_MARKERS = ("cuda out of memory", "cudaerrormemoryallocation", "cuda_error_out_of_memory", "pxgcudadevicememory")


def run_point(n, node_id, seed):
    rec = dict(num_envs=n, ok=False)
    t0 = time.time()
    w = SimWorker.options(
        name=f"phase2_{n}", max_restarts=0, label_selector={"ray.io/node-id": node_id},
        runtime_env={"env_vars": config.ISAAC_ENV_VARS},
    ).remote(0, n, config.TASK, config.TASK_OVERRIDES, seed, config.AGENT_ENTRY, None, config.TELEMETRY_INTERVAL_S)
    try:
        ready, _ = ray.wait([w.ping.remote()], timeout=3600)
        if not ready:
            raise TimeoutError("init > 3600 s")
        ray.get(ready[0])
        rec["driver_init_wall_s"] = time.time() - t0
        ray.get(w.run_random.remote(WARMUP_STEPS), timeout=3600)
        ready, _ = ray.wait([w.run_random.remote(MEASURE_STEPS)], timeout=7200)
        if not ready:
            raise TimeoutError("stepping > 7200 s")
        step = ray.get(ready[0])
        st = ray.get(w.get_stats.remote(), timeout=300)
        tel = st["telemetry"].get("step") or {}
        rec.update(
            ok=True, init_s=st["timings"]["total_init_s"], timings=st["timings"],
            env_steps_per_s=step["env_steps_per_s"], s_per_step=step["seconds"] / step["steps"],
            s_per_step_per_env=step["seconds"] / step["steps"] / n,
            sm_pct_mean=(tel.get("sm_pct") or {}).get("mean"), mem_bw_pct_mean=(tel.get("mem_bw_pct") or {}).get("mean"),
            vram_peak_mib=st["gpu_used_mib"], torch_alloc_mib=st["torch_alloc_mib"],
            torch_reserved_mib=st["torch_reserved_mib"], telemetry=st["telemetry"], episodes_done=step["episodes_done"])
    except Exception as e:  # noqa: BLE001  OOM / PhysX buffer failure / actor death are data points
        msg = f"{type(e).__name__}: {e}"
        rec["error"] = msg[-2000:]
        rec["failure_kind"] = "vram_oom" if any(m in msg.lower() for m in OOM_MARKERS) else "actor_died_check_log"
        rec["driver_wall_s"] = time.time() - t0
    finally:
        try:
            ray.wait([w.close.remote()], timeout=120)
        except Exception:  # noqa: BLE001
            pass
        ray.kill(w, no_restart=True)
        time.sleep(10)  # let the GPU context drain before the next point
    return rec


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot) if ss_tot else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1024)
    ap.add_argument("--max", type=int, default=1 << 20)
    args = ap.parse_args()

    ray.init(address=config.RAY_ADDRESS)
    meta = RunMeta("phase2")
    p1 = json.load(open(os.path.join(config.RESULTS_DIR, "phase1.json")))
    node_id = p1["runs"][0]["placement"]["node_id"]
    host = p1["runs"][0]["placement"]["hostname"]
    seed = config.worker_seed(0)
    meta.note_seed(0, seed)
    print(f"Phase 2 sweep on {host[:24]} ({node_id[:8]}), task {config.TASK} {config.TASK_OVERRIDES}")
    print(f"{'envs':>7} {'ok':>3} {'init_s':>7} {'VRAM GiB':>9} {'env-steps/s':>12} {'s/step':>7} "
          f"{'us/step/env':>11} {'SM%':>5} {'memBW%':>6}  note")

    points, best_per_env, n = [], None, args.start
    while n <= args.max:
        r = run_point(n, node_id, seed)
        points.append(r)
        if not r["ok"]:
            print(f"{n:>7} {'NO':>3}  {r['failure_kind']}: {r['error'][:150]}")
            r["ceiling"] = "vram" if r["failure_kind"] == "vram_oom" else "failure (see worker log)"
            break
        best_per_env = min(best_per_env or 1e9, r["s_per_step_per_env"])
        degraded = r["s_per_step_per_env"] > DEGRADE_FACTOR * best_per_env
        print(f"{n:>7} {'yes':>3} {r['init_s']:7.1f} {r['vram_peak_mib']/1024:9.2f} {r['env_steps_per_s']:12,.0f} "
              f"{r['s_per_step']:7.3f} {1e6*r['s_per_step_per_env']:11.2f} {r['sm_pct_mean'] or -1:5.1f} "
              f"{r['mem_bw_pct_mean'] or -1:6.1f}  {'THROUGHPUT-DEGRADED (>2x best per-env step time)' if degraded else ''}")
        if degraded:
            r["ceiling"] = "throughput"
            break
        if r["vram_peak_mib"] > 0.95 * GPU_TOTAL_MIB:
            r["ceiling"] = "vram"
            print("        VRAM > 95% of GPU: stopping before an OOM point")
            break
        n *= 2

    good = [p for p in points if p["ok"]]
    xs, ys = [p["num_envs"] for p in good], [p["vram_peak_mib"] for p in good]
    a, b, r2 = linfit(xs, ys) if len(good) >= 2 else (ys[0] if ys else 0, 0, 0)
    fixed_gb, per_env_kb = a / 1024, b * 1024
    vram_safe = int((HEADROOM * GPU_TOTAL_MIB - a) / b) if b > 0 else max(xs)
    best_tput = max(p["env_steps_per_s"] for p in good)
    # knee only exists if throughput flattened before the last point; otherwise report "not reached"
    flat = [p for p in good[:-1] if p["env_steps_per_s"] >= KNEE_FRACTION * best_tput]
    knee = min(p["num_envs"] for p in flat) if flat else max(xs)
    envs_per_gpu = max(256, (min(vram_safe, knee) // 256) * 256)
    last_good = max(xs)
    binding = "vram@75%" if vram_safe <= knee else "throughput knee"

    print(f"\nVRAM fit: {fixed_gb:.2f} GB fixed + {per_env_kb:,.0f} KB/env  (R^2 {r2:.4f}, {len(good)} points)")
    print(f"last good point {last_good:,} | stop reason: {points[-1].get('ceiling', 'reached --max')}")
    print(f"VRAM-safe @ {int(HEADROOM*100)}% of 80 GB: {vram_safe:,} envs | throughput knee (>= {int(KNEE_FRACTION*100)}% "
          f"of best {best_tput:,.0f} env-steps/s): {knee:,} envs")
    print(f"ENVS_PER_GPU = min(...) -> {envs_per_gpu:,}  (binding: {binding})")
    print(f"vs AlphaNIST ~{config.PAPER_ENVS_PER_GPU:,}/GPU: {envs_per_gpu/config.PAPER_ENVS_PER_GPU:.2f}x | "
          f"W={config.NUM_WORKERS} aggregate: {config.NUM_WORKERS*envs_per_gpu:,} envs | "
          f"64K needs {-(-config.OMNIRESET_TOTAL_ENVS // envs_per_gpu)} GPUs at this density")

    out = dict(meta=meta.finish(gpu_nodes_used=1), task=config.TASK, overrides=config.TASK_OVERRIDES,
               node=dict(id=node_id, hostname=host), warmup_steps=WARMUP_STEPS, measure_steps=MEASURE_STEPS,
               points=points, fit=dict(fixed_overhead_gb=fixed_gb, per_env_kb=per_env_kb, r2=r2, n_points=len(good)),
               headroom=HEADROOM, vram_safe_envs=vram_safe, throughput_knee_envs=knee, knee_fraction=KNEE_FRACTION,
               best_env_steps_per_s=best_tput, last_good_envs=last_good, binding_ceiling=binding,
               ENVS_PER_GPU=envs_per_gpu, paper_envs_per_gpu=config.PAPER_ENVS_PER_GPU,
               aggregate_at_W=dict(W=config.NUM_WORKERS, envs=config.NUM_WORKERS * envs_per_gpu),
               gpus_for_64k=-(-config.OMNIRESET_TOTAL_ENVS // envs_per_gpu))
    m = out["meta"]
    print(f"sweep wall {m['wall_s']/60:.1f} min | est. ${m['usd_used_nodes']:.2f} (1 GPU node + head) | "
          f"${m['usd_whole_cluster']:.2f} whole cluster for the same time")
    write_json("memory_sweep.json", out)
    _, orphans = ray.get(orphan_check.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)).remote(), timeout=120)
    print(f"orphan Isaac procs on sweep node: {len(orphans)}")


if __name__ == "__main__":
    main()
