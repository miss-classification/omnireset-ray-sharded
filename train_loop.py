"""Driver-side training loop: W SimWorkers + 1 Learner, pipelined rollouts, fault handling.

Driver does orchestration + logging only (ray + stdlib; no isaac, numpy or torch on the head).

Per iteration:
  * every live worker has exactly one collect() in flight; when its buffer arrives the driver immediately
    submits that worker's next collect() (so workers never wait on the learner), then queues the newest
    weights behind it -> each rollout is <= 1 update stale.
  * the learner updates on the buffers that arrived this iteration (ObjectRefs passed as top-level args, so
    data moves worker node -> learner node directly), returns metrics + SGS weights; the driver broadcasts
    weights (one ray.put, shared by all workers) and SGS weights every update.
  * every gather is ray.wait with a timeout; no unbounded ray.get.
Fault handling: a collect() raising RayActorError marks the worker DEAD (event logged); Ray restarts the actor
(max_restarts=-1), whose __init__ rebuilds the sim and reloads its bank shard. The driver polls ping() without
blocking; when it answers, it re-runs setup_training + weights + SGS weights and resubmits collect()
(event REJOINED). Training continues on the surviving workers meanwhile.
"""
import time

import ray
from ray.exceptions import GetTimeoutError, RayActorError

import config
from learner import Learner
from sim_worker import SimWorker


class Timeline:
    def __init__(self):
        self.t0 = time.time()
        self.events = []

    def log(self, it, env_steps, what, **kw):
        e = dict(t=round(time.time() - self.t0, 1), iteration=it, env_steps=env_steps, event=what, **kw)
        self.events.append(e)
        extra = " ".join(f"{k}={v}" for k, v in kw.items())
        print(f"  [t={e['t']:7.1f}s it={it:4d} steps={env_steps:>12,}] {what} {extra}", flush=True)


def selector(extra_avoid=()):
    bad = [n["NodeID"] for n in ray.nodes()
           if any(n["NodeID"].startswith(p) for p in list(config.EXCLUDE_NODE_ID_PREFIXES) + list(extra_avoid))]
    return {"ray.io/node-id": f"!in({','.join(bad)})"} if bad else None


def run(tag, W, envs_per_worker, iterations, *, worker_gpu=1.0, learner_gpu=1.0, kill=None, init_ckpt=None,
        meta=None, extra_avoid=(), post_rejoin_iters=None, worker_gpu_overrides=None):
    """kill: optional (worker_id, at_iteration). iterations is a cap when post_rejoin_iters is set: the run then
    stops that many iterations after the killed worker rejoins. worker_gpu_overrides: {worker_id: num_gpus}
    (Phase 6: worker 0 at 0.5 so the learner at 0.5 packs onto the same GPU). Returns a dict of measurements."""
    worker_gpu_overrides = worker_gpu_overrides or {}
    rejoined_at = None
    sel = selector(extra_avoid)
    tl = Timeline()
    L = config.ROLLOUT_LEN
    renv = {"env_vars": config.ISAAC_ENV_VARS}
    print(f"\n=== {tag}: W={W} x {envs_per_worker:,} envs = {W*envs_per_worker:,} envs | L={L} | "
          f"worker_gpu={worker_gpu} learner_gpu={learner_gpu}")

    # ---- workers: parallel init ----
    t_create = time.time()
    workers = {}
    for w in range(W):
        seed = config.worker_seed(w)
        if meta is not None:
            meta.note_seed(w, seed)
        workers[w] = SimWorker.options(
            name=f"{tag}_w{w}", num_gpus=worker_gpu_overrides.get(w, worker_gpu), label_selector=sel, runtime_env=renv,
        ).remote(w, envs_per_worker, config.TASK, config.TASK_OVERRIDES, seed, config.AGENT_ENTRY,
                 config.BANK_DIR, config.TELEMETRY_INTERVAL_S, config.BANK_PAIR, W)
    pend = {workers[w].ping.remote(): w for w in workers}
    ready_at = {}
    while pend:
        done, _ = ray.wait(list(pend), num_returns=1, timeout=3600)
        if not done:
            raise SystemExit(f"{tag}: workers {sorted(pend.values())} not ready in 3600 s")
        w = pend.pop(done[0])
        ray.get(done[0], timeout=60)
        ready_at[w] = time.time() - t_create
    init_wall = time.time() - t_create
    stats0 = ray.get([workers[w].get_stats.remote() for w in workers], timeout=300)
    init_each = {s["worker_id"]: s["init_seconds"] for s in stats0}
    timings_each = {s["worker_id"]: s["timings"] for s in stats0}
    placement = ray.get([workers[w].placement.remote() for w in workers], timeout=120)

    # ---- training setup ----
    specs = ray.get([workers[w].setup_training.remote(L) for w in workers], timeout=600)
    agent_cfg = ray.get(workers[0].agent_cfg_dict.remote(), timeout=60)
    t = time.time()
    learner = Learner.options(name=f"{tag}_learner", num_gpus=learner_gpu, label_selector=sel).remote(
        agent_cfg, specs[0]["obs_dim"], specs[0]["num_actions"], envs_per_worker, L,
        (config.BANK_DIR, config.BANK_PAIR), f"{config.CKPT_DIR}/{tag}", config.SEED,
        config.SGS_EVERY, config.CKPT_EVERY, init_ckpt, False)
    linfo = ray.get(learner.info.remote(), timeout=600)
    learner_init_s = time.time() - t
    placement.append(dict(worker_id="learner", hostname=linfo["hostname"], node_id=linfo["node_id"],
                          ray_gpu_ids=linfo["ray_gpu_ids"], device_name=linfo["device_name"]))
    print(f"  {'actor':<8} {'hostname':<26} {'gpu':<4} device")
    # (with fractional GPUs, two actors can show the same node + gpu id: that is the point of Phase 6)
    for p in placement:
        print(f"  {str(p['worker_id']):<8} {p['hostname'][:26]:<26} {','.join(p['ray_gpu_ids']):<4} {p['device_name']}")
        assert len(p["ray_gpu_ids"]) >= 1, f"actor {p['worker_id']} has no GPU"
    print(f"  init: wall {init_wall:.1f}s for all {W} workers in parallel | per-worker max {max(init_each.values()):.1f}s "
          f"sum {sum(init_each.values()):.1f}s | bank load (worker 0) {timings_each[0].get('bank_load_s', 0):.2f}s "
          f"| learner {learner_init_s:.1f}s")

    sgs_ids_ref = learner.sgs_ids_list.remote()   # refs only: the head never deserializes numpy/torch
    ver = 0
    wref = learner.get_weights.remote()
    sref = None
    for w in workers:
        workers[w].set_weights.remote(wref, ver)

    # ---- loop ----
    alive = set(workers)
    dead_since = {}          # w -> (ping_ref, t_dead)
    inflight = {workers[w].collect.remote(): w for w in workers}
    iters, env_steps, killed = [], 0, False
    t_loop = time.time()
    for it in range(1, iterations + 1):
        if kill and not killed and it == kill[1]:
            tl.log(it, env_steps, "KILL", worker=kill[0])
            ray.kill(workers[kill[0]], no_restart=False)
            killed = True
        # gather one buffer from every live worker that has one in flight
        t_it = time.time()
        expect = {r: w for r, w in inflight.items() if w in alive}
        got = {}
        while expect:
            done, _ = ray.wait(list(expect), num_returns=1, timeout=900)
            if not done:
                tl.log(it, env_steps, "GATHER_TIMEOUT", waiting_on=sorted(expect.values()))
                break
            r = done[0]
            w = expect.pop(r)
            inflight.pop(r, None)
            try:
                st = ray.get(r, timeout=0)     # small python dict + nested ObjectRef to the buffer
            except RayActorError as e:
                alive.discard(w)
                dead_since[w] = (workers[w].ping.remote(), time.time())
                tl.log(it, env_steps, "WORKER_DIED", worker=w, error=type(e).__name__)
                continue
            got[w] = st
            inflight[workers[w].collect.remote()] = w   # next rollout starts now (staleness <= 1)
        if not got:
            tl.log(it, env_steps, "NO_BUFFERS")
            continue
        t_gather = time.time() - t_it
        # learner update on whatever arrived
        try:
            rec = ray.get(learner.update.remote(*[got[w]["buf"] for w in sorted(got)]), timeout=1800)
        except GetTimeoutError:
            tl.log(it, env_steps, "LEARNER_TIMEOUT")
            continue
        ver = rec["version"]
        wref = learner.get_weights.remote()
        if rec["new_sgs_weights"]:
            sref = learner.get_sgs_weights.remote()
        for w in alive:
            workers[w].set_weights.remote(wref, ver)
            if sref is not None:
                workers[w].set_sgs_weights.remote(sgs_ids_ref, sref)
        bstats = [got[w] for w in sorted(got)]
        steps_it = sum(b["envs"] * L for b in bstats)
        env_steps += steps_it
        it_wall = time.time() - t_it
        row = dict(iteration=it, workers=sorted(got), env_steps=steps_it, wall_s=it_wall, gather_s=t_gather,
                   agg_env_steps_per_s=steps_it / it_wall,
                   per_worker_env_steps_per_s={b["worker_id"]: b["env_steps_per_s"] for b in bstats},
                   buffer_mb=sum(b["buffer_mb"] for b in bstats), learner=rec,
                   episodes=sum(b["episodes"] for b in bstats),
                   success=sum(b["success"] or 0 for b in bstats), attempts=sum(b["attempts"] or 0 for b in bstats),
                   mean_ep_return=_mean([b["mean_ep_return"] for b in bstats]))
        iters.append(row)
        sr = row["success"] / row["attempts"] if row["attempts"] else float("nan")
        print(f"  it {it:3d} | {len(got)}/{W} workers | {row['agg_env_steps_per_s']:>10,.0f} env-steps/s | "
              f"wall {it_wall:5.1f}s (learner update {rec['update_s']:.2f}s, ingest {rec['ingest_s']:.2f}s, "
              f"sgs {1e3*rec['sgs_merge_s']:.1f}ms) | buf {row['buffer_mb']:.0f} MB | succ {sr:.3f} "
              f"| ret {row['mean_ep_return'] if row['mean_ep_return'] is None else round(row['mean_ep_return'], 2)}",
              flush=True)
        # rejoin restarted workers (non-blocking)
        for w, (pref, t_dead) in list(dead_since.items()):
            ok, _ = ray.wait([pref], timeout=0)
            if not ok:
                continue
            try:
                ray.get(ok[0], timeout=0)
            except RayActorError:
                dead_since[w] = (workers[w].ping.remote(), t_dead)  # still restarting; poll again
                continue
            ray.get(workers[w].setup_training.remote(L), timeout=600)
            ray.get(workers[w].set_weights.remote(wref, ver), timeout=300)
            if sref is not None:
                workers[w].set_sgs_weights.remote(sgs_ids_ref, sref)
            st = ray.get(workers[w].get_stats.remote(), timeout=120)
            alive.add(w)
            del dead_since[w]
            inflight[workers[w].collect.remote()] = w
            tl.log(it, env_steps, "REJOINED", worker=w, down_s=round(time.time() - t_dead, 1),
                   reinit_s=round(st["init_seconds"], 1), bank_load_s=round(st["timings"].get("bank_load_s", 0), 2))
            rejoined_at = it
        if post_rejoin_iters is not None and rejoined_at is not None and it >= rejoined_at + post_rejoin_iters:
            tl.log(it, env_steps, "STOP", reason=f"{post_rejoin_iters} iterations after rejoin")
            break

    loop_wall = time.time() - t_loop
    sgs_final = ray.get(learner.sgs_summary.remote(), timeout=120)
    final_stats = {}
    for w in list(alive):
        try:
            final_stats[w] = ray.get(workers[w].get_stats.remote(), timeout=120)
        except Exception:  # noqa: BLE001
            pass
    # shutdown: close + hard kill
    closes = [workers[w].close.remote() for w in alive]
    ray.wait(closes, num_returns=len(closes), timeout=120)
    for h in list(workers.values()) + [learner]:
        ray.kill(h, no_restart=True)
    return dict(tag=tag, W=W, envs_per_worker=envs_per_worker, total_envs=W * envs_per_worker, L=L,
                worker_gpu=worker_gpu, learner_gpu=learner_gpu, placement=placement,
                init_wall_s=init_wall, init_each_s=init_each, init_ready_at_s=ready_at, timings_each=timings_each,
                learner_init_s=learner_init_s, iterations=iters, loop_wall_s=loop_wall, timeline=tl.events,
                sgs=sgs_final,
                final_worker_stats={str(k): dict(telemetry=v["telemetry"], gpu_used_mib=v["gpu_used_mib"])
                                    for k, v in final_stats.items()})


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
