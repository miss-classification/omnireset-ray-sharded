"""Host RAM per env + where the 64K PhysX material ceiling sits.

Customer: a single 8-GPU node with 2 TB RAM gets its process killed at ~500K envs (host RAM, not VRAM), and one
PhysX process caps out at 64K materials. memory_sweep.py measures VRAM. Here each point is one fresh SimWorker
(OmniReset peg task, original reset-data path), all points in parallel, one per
healthy GPU node (each process has its own node, so RSS is per process and node RAM is not shared).

Per point: actor RSS / high-water mark after init and after stepping, RSS split (anon / file / shmem), node RAM,
VRAM, USD physics-material prims per env. 24,576 envs sits between the last good (16,384) and the crash (32,768)
point of the VRAM sweep: it bounds the per-process material ceiling.

Output: host_ram.json with RSS = a + b*N fit and extrapolations (500K and 1M envs; per node of 8 processes).
"""
import argparse
import time

import ray

import config
from common import RunMeta, write_json
from memory_sweep import OOM_MARKERS, linfit
from sim_worker import SimWorker

POINTS = [1024, 2048, 4096, 8192, 16384, 24576]
WARMUP_STEPS, MEASURE_STEPS = 10, 60
NODE_RAM_A2_ULTRAGPU_1G_GIB = 170  # GCP spec; measured MemTotal is recorded per point too


def start(n, wid, sel):
    seed = config.worker_seed(wid)
    a = SimWorker.options(name=f"p2b_{n}", max_restarts=0, num_gpus=1, label_selector=sel,
                          runtime_env={"env_vars": config.ISAAC_ENV_VARS}) \
        .remote(wid, n, config.TASK, config.TASK_OVERRIDES, seed, config.AGENT_ENTRY, None, config.TELEMETRY_INTERVAL_S)
    return a, seed


def measure(n, a, t0):
    rec = dict(num_envs=n, ok=False)
    try:
        ray.get(a.ping.remote(), timeout=3600)
        rec["init_wall_s"] = time.time() - t0
        rec["placement"] = ray.get(a.placement.remote(), timeout=120)
        rec["mem_after_init"] = ray.get(a.host_mem.remote(), timeout=120)
        try:
            rec["materials"] = ray.get(a.count_physics_materials.remote(), timeout=600)
        except Exception as e:  # noqa: BLE001
            rec["materials"] = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
        ray.get(a.run_random.remote(WARMUP_STEPS), timeout=3600)
        step = ray.get(a.run_random.remote(MEASURE_STEPS), timeout=7200)
        rec["mem_after_step"] = ray.get(a.host_mem.remote(), timeout=120)
        st = ray.get(a.get_stats.remote(), timeout=300)
        rec.update(ok=True, env_steps_per_s=step["env_steps_per_s"], vram_peak_mib=st["gpu_used_mib"],
                   init_s=st["timings"]["total_init_s"], timings=st["timings"])
    except Exception as e:  # noqa: BLE001  a crash is a data point
        msg = f"{type(e).__name__}: {e}"
        rec["error"] = msg[-1500:]
        rec["failure_kind"] = "vram_oom" if any(m in msg.lower() for m in OOM_MARKERS) else "actor_died_check_log"
        rec["wall_s"] = time.time() - t0
    finally:
        try:
            ray.wait([a.close.remote()], timeout=120)
        except Exception:  # noqa: BLE001
            pass
        ray.kill(a, no_restart=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, nargs="*", default=POINTS)
    args = ap.parse_args()
    ray.init(address=config.RAY_ADDRESS)
    meta = RunMeta("phase2b")
    sel = config.actor_label_selector(ray.nodes())
    t0 = time.time()
    actors = []
    for i, n in enumerate(args.points):
        a, seed = start(n, i, sel)
        meta.note_seed(i, seed)
        actors.append((n, a))
    import concurrent.futures as cf  # driver threads only wait on refs (every ray.get has a timeout)
    with cf.ThreadPoolExecutor(len(actors)) as ex:
        recs = list(ex.map(lambda na: measure(na[0], na[1], t0), actors))
    recs.sort(key=lambda r: r["num_envs"])

    print(f"\n{'envs':>7} {'ok':>3} {'RSS GiB':>8} {'HWM GiB':>8} {'anon':>6} {'file':>6} {'shmem':>6} "
          f"{'VRAM GiB':>8} {'mat/env':>7} {'mat total':>9} {'env-steps/s':>11}  note")
    for r in recs:
        m = r.get("mem_after_step") or r.get("mem_after_init") or {}
        mt = r.get("materials") or {}
        g = lambda k: (m.get(k) or 0) / 1024  # noqa: E731
        print(f"{r['num_envs']:>7,} {('yes' if r['ok'] else 'NO'):>3} {g('rss_mib'):>8.2f} {g('hwm_mib'):>8.2f} "
              f"{g('rss_anon_mib'):>6.1f} {g('rss_file_mib'):>6.1f} {g('rss_shmem_mib'):>6.1f} "
              f"{(r.get('vram_peak_mib') or 0) / 1024:>8.2f} {str(mt.get('per_env', '-')):>7} "
              f"{str(mt.get('usd_total_est', '-')):>9} {r.get('env_steps_per_s', 0):>11,.0f}  {r.get('error', '')[:90]}")

    ok = [r for r in recs if r["ok"]]
    fit = {}
    if len(ok) >= 2:
        xs = [r["num_envs"] for r in ok]
        for key in ("hwm_mib", "rss_mib"):
            ys = [r["mem_after_step"][key] for r in ok]
            a, b, r2 = linfit(xs, ys)
            fit[key] = dict(fixed_gib=a / 1024, per_env_kib=b * 1024, r2=r2)
        f = fit["hwm_mib"]
        per_env_gib = f["per_env_kib"] / 2**20

        def node_gib(total_envs, procs):
            return procs * f["fixed_gib"] + total_envs * per_env_gib
        fit["extrapolation"] = {
            "500K_envs_1_node_8_procs_gib": node_gib(500_000, 8),
            "500K_envs_1_node_1_proc_gib": node_gib(500_000, 1),
            "1M_envs_1_node_8_procs_gib": node_gib(1_000_000, 8),
            "16384_envs_1_proc_gib": node_gib(16384, 1),
            "envs_per_2TB_node_8_procs": int((2048 - 8 * f["fixed_gib"]) / per_env_gib),
            "envs_per_170GB_a2_node_1_proc": int((NODE_RAM_A2_ULTRAGPU_1G_GIB * 0.9 - f["fixed_gib"]) / per_env_gib),
        }
        print(f"\nhost RAM (high-water) = {f['fixed_gib']:.2f} GiB + {f['per_env_kib']:.1f} KiB/env per process "
              f"(R^2 {f['r2']:.4f})")
        for k, v in fit["extrapolation"].items():
            print(f"  {k}: {v:,.0f}" if isinstance(v, int) else f"  {k}: {v:,.1f}")
    out = dict(meta=meta.finish(gpu_nodes_used=len(args.points)), points=recs, fit=fit,
               note="RSS/HWM of the actor process (Isaac Sim runs in-process); children summed separately.")
    m = out["meta"]
    print(f"wall {m['wall_s']/60:.1f} min | est. ${m['usd_used_nodes']:.2f} | ${m['usd_whole_cluster']:.2f} whole cluster")
    write_json("host_ram.json", out)


if __name__ == "__main__":
    main()
