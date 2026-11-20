"""SimWorker: 1 Ray actor = 1 process = 1 GPU = 1 Isaac Sim instance + 1 vectorized Isaac Lab env.

Rules this file obeys:
  * No isaacsim / isaaclab / omni / numpy / torch import at module level. The driver imports this module on
    the head node (CPU-only, no numpy/torch); Kit must only ever boot inside the actor, on a GPU node.
  * The simulation app is created before any isaaclab env/task import, once per process.
  * Ray sets CUDA_VISIBLE_DEVICES per actor; we never override it and always use cuda:0.
  * Everything leaving the actor is a batched array / small dict, never per-env objects.

API notes (Isaac Lab 3.0.0-EA @ ae37b02 + UWLab 2b40afb, read from installed source):
  * App launch: isaaclab.app.AppLauncher(dict) as in UWLab scripts/reinforcement_learning/rsl_rl/train.py.
  * Task config: uwlab_tasks.utils.hydra.hydra_task_compose(task, agent_entry, overrides) -- the
    programmatic twin of the @hydra_task_config decorator train.py uses; no sys.argv involved.
  * Env: gym.make(task, cfg=env_cfg) -> ManagerBasedRLEnv. Per-env auto-reset is built in
    (ManagerBasedRLEnv.step resets done env_ids through the event manager's "reset" terms, which for
    OmniReset is MultiResetManager sampling from the released reset-state datasets). No global reset.

Init breakdown (measured, not inferred): MultiResetManager.__init__ is wrapped, and inside that window
utils.safe_retrieve_file_path (download / cache lookup), torch.load (deserialize) and sample_state_data_set
(the per-element Python rebuild loop) are timed separately. gym.make time is reported both including and
excluding the MultiResetManager window.
"""
import os
import socket
import subprocess
import sys
import time

import ray


@ray.remote(num_gpus=1, max_restarts=-1, max_task_retries=0)
class SimWorker:
    def __init__(self, worker_id: int, num_envs: int, task: str, overrides: list, seed: int,
                 agent_entry: str = "rsl_rl_cfg_entry_point", reset_bank_path: str | None = None,
                 telemetry_interval_s: int = 2, bank_pair: str = "Peg__PegHole", num_workers: int = 1,
                 cfg_patch: str | None = None, profile_init: bool = False):
        t0 = time.time()
        self.worker_id, self.num_envs, self.task, self.seed = worker_id, num_envs, task, seed
        self.timings = {}

        from gpu_telemetry import GpuTelemetry
        self.tel = GpuTelemetry(telemetry_interval_s)
        self.tel.mark("init")

        # 0. self-provision (no-op on a provisioned node; a fresh replacement node installs here)
        import cluster_env
        self.timings["provision_s"] = cluster_env.ensure_stack()["seconds"]

        # 1. boot Kit, headless. Hide Ray's worker argv from Kit's own arg parser.
        t = time.time()
        argv, sys.argv = sys.argv, [sys.argv[0]]
        try:
            from isaaclab.app import AppLauncher
            self._launcher = AppLauncher({"headless": True, "device": "cuda:0"})
            self.app = self._launcher.app
        finally:
            sys.argv = argv
        self.timings["app_boot_s"] = time.time() - t

        # 2. task registration + config compose (task imports only after the app exists)
        t = time.time()
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401  registers Isaac Lab tasks
        import uwlab_tasks  # noqa: F401     registers OmniReset tasks
        from uwlab_tasks.utils.hydra import hydra_task_compose
        self.timings["task_import_s"] = time.time() - t

        t = time.time()

        @hydra_task_compose(task, agent_entry, list(overrides))
        def _cfg(env_cfg, agent_cfg):
            return env_cfg, agent_cfg

        env_cfg, agent_cfg = _cfg()
        # adapt the agent cfg to the installed rsl-rl (UW fork 5.4.1), exactly as UWLab's play/train scripts do
        sys.path.insert(0, "/home/ray/UWLab/scripts/reinforcement_learning/rsl_rl")
        import cli_args
        self.agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = seed
        env_cfg.sim.device = "cuda:0"
        self.patch_info = None
        if cfg_patch:  # experiment hooks (scene_patches.py), e.g. shared physics materials
            import scene_patches
            self.patch_info = getattr(scene_patches, cfg_patch)(env_cfg)
        self.timings["task_compose_s"] = time.time() - t

        # 3. gym.make. Reset states come either from OmniReset's own loader (instrumented) or, with a cooked
        #    bank, from this worker's parquet shard (bank mode also enables per-reset-state SGS counters).
        self.bank = None
        if reset_bank_path:
            self._install_bank_reset(reset_bank_path, bank_pair, worker_id, num_workers)
        else:
            self._instrument_multireset()
        self._instrument_env_stages()
        self.profile = None
        prof = self._start_profiler() if profile_init else None
        t = time.time()
        self.env = gym.make(task, cfg=env_cfg)
        self.timings["gym_make_s"] = time.time() - t
        if prof:
            self.profile = self._stop_profiler(prof)
        reset_data_s = self.timings.get("mrm_total_s", self.timings.get("bank_load_s", 0.0))
        self.timings["gym_make_excl_reset_datasets_s"] = self.timings["gym_make_s"] - reset_data_s

        t = time.time()
        self.uenv = self.env.unwrapped
        self.obs, _ = self.env.reset()
        torch.cuda.synchronize()
        self.timings["first_reset_s"] = time.time() - t

        self.torch = torch
        self.device = self.uenv.device
        self.act_dim = self.uenv.action_manager.total_action_dim
        self.init_seconds = time.time() - t0
        self.timings["total_init_s"] = self.init_seconds
        self.timings["other_s"] = self.init_seconds - sum(
            self.timings[k] for k in ("provision_s", "app_boot_s", "task_import_s", "task_compose_s",
                                      "gym_make_s", "first_reset_s"))
        self.total_steps = 0
        self.step_time = 0.0
        self.alg = None
        self.tel.mark("idle")

    # ---------------- gym.make internals: scene creation / physics start / managers ----------------
    def _instrument_env_stages(self):
        """Time Isaac Lab's own init stages (ManagerBasedEnv.__init__ order): InteractiveScene (USD spawn + clone),
        SimulationContext.reset (physics parse + play; class-term inits that run on play), load_managers."""
        from isaaclab.envs import manager_based_env as mbe
        from isaaclab.scene import interactive_scene as isc
        import isaaclab.sim as isim
        T = self.timings

        def wrap(owner, name, key):
            orig = getattr(owner, name)
            if getattr(orig, "_timed", False):
                return

            def f(*a, **k):
                t = time.time()
                try:
                    return orig(*a, **k)
                finally:
                    T[key] = T.get(key, 0.0) + time.time() - t
            f._timed = True
            setattr(owner, name, f)
        wrap(isc.InteractiveScene, "__init__", "stage_scene_creation_s")
        wrap(isim.SimulationContext, "reset", "stage_sim_reset_s")
        wrap(mbe.ManagerBasedEnv, "load_managers", "stage_load_managers_base_s")
        from isaaclab.envs import manager_based_rl_env as mbr
        if "load_managers" in mbr.ManagerBasedRLEnv.__dict__:  # RL override (inclusive of the base call)
            wrap(mbr.ManagerBasedRLEnv, "load_managers", "stage_load_managers_s")

    def _start_profiler(self, rate=50):
        """py-spy sampling of this actor's own process during gym.make (Python frames + native)."""
        out = f"/tmp/pyspy_{os.getpid()}.txt"
        base = ["py-spy", "record", "--pid", str(os.getpid()), "--rate", str(rate), "--format", "raw",
                "--output", out, "--nonblocking"]
        base[0] = os.path.join(os.path.dirname(sys.executable), "py-spy")  # sudo resets PATH
        # no --native: native unwinding of Kit stalled the process (2026-11-20 smoke test)
        for cmd in (["sudo", "-n"] + base, base):
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(3)
            if p.poll() is None:
                return dict(proc=p, out=out, cmd=" ".join(cmd))
        return dict(proc=None, out=None, cmd=None, error=p.stderr.read().decode()[-500:])

    def _stop_profiler(self, prof):
        import collections
        import signal
        if not prof.get("proc"):
            return dict(error=prof.get("error"))
        prof["proc"].send_signal(signal.SIGINT)
        try:
            prof["proc"].wait(timeout=120)
        except subprocess.TimeoutExpired:
            prof["proc"].kill()
        if not os.path.exists(prof["out"]):
            return dict(error="no profile output", cmd=prof["cmd"], stderr=prof["proc"].stderr.read().decode()[-500:])
        leaf, incl, total = collections.Counter(), collections.Counter(), 0
        for line in open(prof["out"]):
            stack, _, n = line.rstrip().rpartition(" ")
            if not n.isdigit():
                continue
            n = int(n)
            frames = stack.split(";")
            total += n
            leaf[frames[-1][:160]] += n
            for fr in set(frames):
                incl[fr[:160]] += n
        import config
        dst = os.path.join(config.RESULTS_DIR, "profiles", f"init_{self.num_envs}_{os.getpid()}.txt")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(["cp", prof["out"], dst])
        return dict(cmd=prof["cmd"], samples=total, file=dst, top_self=leaf.most_common(40),
                    top_inclusive=incl.most_common(80))

    # ---------------- reset bank + SGS ----------------
    def _install_bank_reset(self, bank_dir, pair, worker_id, num_workers):
        """Replace OmniReset's MultiResetManager loading + sampling with this worker's cooked bank shard.

        Same reset semantics as upstream (_reset_to with env-relative poses, zero joint-velocity targets), but:
          * init reads 1/num_workers of the bank with one columnar parquet read (no torch.load of pickled lists);
          * each auto-reset samples a *reset state* (not a reset type) from SGS weights over the shard;
          * on every reset of a finished episode, the outcome (success flag of the reward term, the same signal
            upstream's SuccessMonitor uses) is credited to the state that episode started from:
            local counters success[i], attempts[i] on device. get_sgs_deltas() returns and zeroes them.
        """
        import torch
        from isaaclab.managers import ManagerTermBase
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp import events as ev

        import bank_pipeline as bp
        worker, T = self, self.timings

        def init(this, cfg, env):
            ManagerTermBase.__init__(this, cfg, env)
            t = time.time()
            state, ids, tid, tm = bp.load_worker_shard(bank_dir, pair, worker_id, num_workers, device=str(env.device))
            T.update(bank_load_s=time.time() - t, bank_rows=tm["rows"], bank_files=tm["files"],
                     bank_read_s=tm["read_s"], bank_to_tensor_s=tm["to_tensor_s"])
            n = len(ids)
            this.bank_state, this.bank_ids, this.bank_tid = state, ids, tid
            this.sgs_w = torch.ones(n, device=env.device)          # sampling weights (uniform until learner sends)
            this.sgs_succ = torch.zeros(n, device=env.device)
            this.sgs_att = torch.zeros(n, device=env.device)
            this.cur = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
            this.success_expr = cfg.params.get("success")
            worker.bank = this

        def call(this, env, env_ids, dataset_dir=None, reset_types=None, probs=None, success=None):
            if env_ids is None:
                env_ids = torch.arange(env.num_envs, device=env.device)
            started = this.cur[env_ids]
            live = started >= 0  # first reset of an env has no finished episode to credit
            if success is not None and bool(live.any()):
                s = eval(success)[env_ids].float()  # noqa: S307  same expression upstream evaluates
                idx = started[live]
                this.sgs_att.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
                this.sgs_succ.index_add_(0, idx, s[live])
            new = torch.multinomial(this.sgs_w, len(env_ids), replacement=True)
            this.cur[env_ids] = new
            st = {a: {b: {f: v[new] for f, v in fs.items()} for b, fs in bs.items()} for a, bs in this.bank_state.items()}
            this._reset_to(st, env_ids=env_ids, is_relative=True)
            robot = env.scene["robot"]
            robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)

        ev.MultiResetManager.__init__ = init
        ev.MultiResetManager.__call__ = call

    def get_sgs_deltas(self):
        """(reset_state_id, successes, attempts) for states touched since the last call; zeroes local deltas."""
        b = self.bank
        touched = (b.sgs_att > 0).nonzero().squeeze(1)
        out = dict(ids=b.bank_ids[touched].cpu().numpy(), succ=b.sgs_succ[touched].cpu().numpy().astype("float32"),
                   att=b.sgs_att[touched].cpu().numpy().astype("float32"))
        b.sgs_succ.zero_(); b.sgs_att.zero_()
        return out

    def set_sgs_weights(self, all_ids, all_w):
        """Learner broadcast: weights for every bank state of the pair; keep this worker's subset."""
        torch, b = self.torch, self.bank
        ids = torch.as_tensor(all_ids, device=self.device)
        w = torch.as_tensor(all_w, device=self.device, dtype=torch.float)
        pos = torch.searchsorted(ids, b.bank_ids)
        b.sgs_w = w[pos.clamp(max=len(ids) - 1)].clamp(min=1e-6)
        return float(b.sgs_w.sum())

    # ---------------- training (rsl_rl PPO rollouts, local policy copy) ----------------
    def agent_cfg_dict(self):
        return self.agent_cfg.to_dict()

    def setup_training(self, rollout_len: int):
        """Local rsl_rl PPO instance used only for act()/process_env_step() into its RolloutStorage (len L).
        The learner owns optimization; this copy receives its weights (actor/critic incl. normalizers)."""
        import copy
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.algorithms import PPO
        cfg = copy.deepcopy(self.agent_cfg.to_dict())
        cfg["num_steps_per_env"] = rollout_len
        cfg["multi_gpu"] = None
        self.L = rollout_len
        self.venv = RslRlVecEnvWrapper(self.env, clip_actions=self.agent_cfg.clip_actions)
        self.vobs = self.venv.get_observations().to(self.device)
        self.alg = PPO.construct_algorithm(self.vobs, self.venv, cfg, self.device)
        self.alg.train_mode()
        self.ep_ret = self.torch.zeros(self.num_envs, device=self.device)
        self.weights_version = -1
        from ray.util.metrics import Gauge
        self._g_tput = Gauge("sim_env_steps_per_s", "env-steps/s of the last rollout", tag_keys=("worker",))
        self._g_sm = Gauge("sim_gpu_sm_pct", "GPU SM% (dmon) during the last rollout", tag_keys=("worker",))
        return dict(obs_dim=int(self.vobs["policy"].shape[1]), num_actions=int(self.venv.num_actions),
                    num_envs=self.num_envs, obs_groups=list(self.vobs.keys()))

    def set_weights(self, weights, version):
        torch = self.torch
        with torch.no_grad():
            self.alg.load(weights, {"actor": True, "critic": True}, strict=True)
        self.weights_version = version
        return version

    def collect(self):
        """L steps with the local policy -> ONE batched rollout buffer (numpy) + SGS deltas + stats."""
        torch = self.torch
        self.tel.mark("step")
        torch.cuda.synchronize()
        t = time.time()
        finished = []
        with torch.no_grad():
            for _ in range(self.L):
                actions = self.alg.act(self.vobs)
                obs, rew, dones, extras = self.venv.step(actions.to(self.venv.device))
                obs, rew, dones = obs.to(self.device), rew.to(self.device), dones.to(self.device)
                self.alg.process_env_step(obs, rew, dones, extras)
                self.ep_ret += rew
                d = dones.bool()
                if bool(d.any()):
                    finished.append(self.ep_ret[d].clone())
                    self.ep_ret[d] = 0
                self.vobs = obs
        torch.cuda.synchronize()
        dt = time.time() - t
        self.tel.mark("idle")
        st = self.alg.storage
        mu, sigma = st.distribution_params[:2]
        buf = dict(worker_id=self.worker_id, weights_version=self.weights_version,
                   obs=st.observations["policy"].cpu().numpy(), actions=st.actions.cpu().numpy(),
                   rewards=st.rewards.cpu().numpy(), dones=st.dones.cpu().numpy().astype("uint8"),
                   values=st.values.cpu().numpy(), logp=st.actions_log_prob.cpu().numpy(),
                   mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy(), last_obs=self.vobs["policy"].cpu().numpy())
        st.clear()
        sgs = self.get_sgs_deltas() if self.bank is not None else None
        fin = torch.cat(finished) if finished else torch.zeros(0, device=self.device)
        self.total_steps += self.L
        self.step_time += dt
        tput = self.num_envs * self.L / dt
        tel = self.tel.summary().get("step") or {}
        sm = (tel.get("sm_pct") or {}).get("mean")
        self._g_tput.set(tput, tags={"worker": str(self.worker_id)})
        if sm is not None:
            self._g_sm.set(sm, tags={"worker": str(self.worker_id)})
        buf["sgs"] = sgs
        stats = dict(worker_id=self.worker_id, weights_version=self.weights_version, envs=self.num_envs,
                     rollout_s=dt, env_steps_per_s=tput, episodes=int(len(fin)),
                     mean_ep_return=float(fin.mean()) if len(fin) else None,
                     success=float(sgs["succ"].sum()) if sgs is not None else None,
                     attempts=float(sgs["att"].sum()) if sgs is not None else None,
                     sgs_delta_states=int(len(sgs["ids"])) if sgs is not None else 0,
                     buffer_mb=sum(v.nbytes for v in buf.values() if hasattr(v, "nbytes")) / 2**20)
        # The driver runs on a head node without numpy/torch: it only ever sees this small python dict plus an
        # ObjectRef to the buffer, which it hands to the learner (Ray moves it worker node -> learner node).
        stats["buf"] = ray.put(buf)
        return stats

    # ---------------- instrumentation ----------------
    def _instrument_multireset(self):
        """Time OmniReset's MultiResetManager.__init__ and its three loading stages, without editing UWLab."""
        import torch
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp import events as ev

        T = self.timings
        T.update(mrm_total_s=0.0, mrm_download_s=0.0, mrm_torch_load_s=0.0, mrm_rebuild_loop_s=0.0,
                 mrm_files=0, mrm_states=0, mrm_bytes=0)
        orig_init, orig_retr, orig_rebuild = ev.MultiResetManager.__init__, ev.utils.safe_retrieve_file_path, \
            ev.sample_state_data_set
        orig_load = torch.load

        def retr(*a, **k):
            t = time.time()
            p = orig_retr(*a, **k)
            T["mrm_download_s"] += time.time() - t
            T["mrm_files"] += 1
            try:
                T["mrm_bytes"] += os.path.getsize(p)
            except OSError:
                pass
            return p

        def load(*a, **k):
            t = time.time()
            r = orig_load(*a, **k)
            T["mrm_torch_load_s"] += time.time() - t
            return r

        depth = [0]  # sample_state_data_set recurses into nested dicts via its module-global name, i.e. through
        #              this wrapper: time/count only the outermost call (before 2026-11-19 23:4x this counted every
        #              recursion level: mrm_states was 9x too high and mrm_rebuild_loop_s an upper bound)

        def rebuild(episode_data, idx, device):
            if depth[0]:
                return orig_rebuild(episode_data, idx, device)
            depth[0] += 1
            t = time.time()
            try:
                r = orig_rebuild(episode_data, idx, device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            finally:
                depth[0] -= 1
            T["mrm_rebuild_loop_s"] += time.time() - t
            T["mrm_states"] += int(len(idx))
            return r

        def init(this, cfg, env):
            t = time.time()
            ev.utils.safe_retrieve_file_path, ev.sample_state_data_set, torch.load = retr, rebuild, load
            try:
                orig_init(this, cfg, env)
            finally:
                ev.utils.safe_retrieve_file_path, ev.sample_state_data_set, torch.load = \
                    orig_retr, orig_rebuild, orig_load
                T["mrm_total_s"] += time.time() - t

        ev.MultiResetManager.__init__ = init

    # ---------------- placement / health ----------------
    def placement(self):
        import torch
        return dict(worker_id=self.worker_id, seed=self.seed, hostname=socket.gethostname(),
                    node_id=ray.get_runtime_context().get_node_id(),
                    ray_gpu_ids=[str(g) for g in ray.get_gpu_ids()],
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    device_name=torch.cuda.get_device_name(0), pid=os.getpid())

    def get_profile(self):
        return self.profile

    def ping(self):
        return self.worker_id

    # ---------------- stepping ----------------
    def run_random(self, n_steps: int):
        """Smoke test: n_steps of uniform random actions, stepped locally (no per-step RPC)."""
        torch = self.torch
        dones_total = 0
        self.tel.mark("step")
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n_steps):
            a = 2.0 * torch.rand((self.num_envs, self.act_dim), device=self.device) - 1.0
            self.obs, rew, term, trunc, _ = self.env.step(a)
            dones_total += int((term | trunc).sum().item())
        torch.cuda.synchronize()
        dt = time.time() - t
        self.tel.mark("idle")
        self.total_steps += n_steps
        self.step_time += dt
        return dict(steps=n_steps, seconds=dt, env_steps_per_s=self.num_envs * n_steps / dt,
                    iters_per_s=n_steps / dt, episodes_done=dones_total)

    # ---------------- host memory / materials ----------------
    def host_mem(self):
        """Host RAM of this actor process (Kit runs in-process) + its children, and node-level totals. Linux /proc only."""
        def status(pid):
            d = {}
            try:
                for line in open(f"/proc/{pid}/status"):
                    k, _, v = line.partition(":")
                    if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"):
                        d[k] = int(v.split()[0]) / 1024  # MiB
            except OSError:
                pass
            return d
        me = os.getpid()
        kids = []
        try:
            kids = [int(p) for p in open(f"/proc/{me}/task/{me}/children").read().split()]
        except OSError:
            pass
        mi = {}
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable", "Shmem"):
                mi[k] = int(v.split()[0]) / 1024
        s = status(me)
        return dict(rss_mib=s.get("VmRSS"), hwm_mib=s.get("VmHWM"), rss_anon_mib=s.get("RssAnon"),
                    rss_file_mib=s.get("RssFile"), rss_shmem_mib=s.get("RssShmem"),
                    children_rss_mib=sum(status(k).get("VmRSS", 0) for k in kids), n_children=len(kids),
                    node_mem_total_mib=mi.get("MemTotal"), node_mem_available_mib=mi.get("MemAvailable"),
                    node_shmem_mib=mi.get("Shmem"))

    def count_physics_materials(self):
        """USD prims with UsdPhysics.MaterialAPI: in env_0 (x num_envs if cloned per env) and outside /World/envs."""
        import omni.usd
        from pxr import Usd, UsdPhysics
        stage = omni.usd.get_context().get_stage()

        def count(root):
            prim = stage.GetPrimAtPath(root)
            if not prim or not prim.IsValid():
                return 0, []
            n, paths = 0, []
            for p in Usd.PrimRange(prim):
                if p.HasAPI(UsdPhysics.MaterialAPI):
                    n += 1
                    paths.append(str(p.GetPath()))
            return n, paths
        per_env, env_paths = count("/World/envs/env_0")
        glob, glob_paths = 0, []
        world = stage.GetPrimAtPath("/World")
        for child in (world.GetChildren() if world else []):
            if child.GetName() == "envs":
                continue
            n, p = count(str(child.GetPath()))
            glob += n
            glob_paths += p
        for child in stage.GetPseudoRoot().GetChildren():  # materials may also live outside /World
            if child.GetName() == "World":
                continue
            n, p = count(str(child.GetPath()))
            glob += n
            glob_paths += p
        return dict(per_env=per_env, env0_paths=env_paths[:20], global_=glob, global_paths=glob_paths[:20],
                    usd_total_est=per_env * self.num_envs + glob)

    # ---------------- stats / shutdown ----------------
    def get_stats(self):
        torch = self.torch
        tel = self.tel.summary()
        return dict(worker_id=self.worker_id, seed=self.seed, num_envs=self.num_envs,
                    init_seconds=self.init_seconds, timings=self.timings, total_steps=self.total_steps,
                    env_steps_per_s=(self.num_envs * self.total_steps / self.step_time) if self.step_time else 0.0,
                    torch_alloc_mib=torch.cuda.memory_allocated() / 2**20,
                    torch_reserved_mib=torch.cuda.memory_reserved() / 2**20,
                    gpu_used_mib=self._peak_vram(tel),
                    telemetry=tel,
                    physics_dt=float(self.uenv.physics_dt), step_dt=float(self.uenv.step_dt),
                    act_dim=int(self.act_dim))

    @staticmethod
    def _peak_vram(tel):
        peaks = [w["vram_mib"]["max"] for k, w in tel.items() if not k.startswith("_") and w.get("vram_mib")]
        return max(peaks) if peaks else -1

    def close(self):
        self.tel.stop()
        try:
            self.env.close()
        finally:
            self.app.close()
        return True


# Isaac/Kit processes only: Ray actor processes for our classes, the kit binary, or python running isaacsim.
# (A bare 'kit' match catches unrelated daemons such as polkitd/packagekitd.)
ORPHAN_PATTERN = r"ray::SimWorker|ray::Learner|/kit/kit|kit/python|isaacsim|omni\.kit"


@ray.remote(num_cpus=0)  # 0 so it also schedules on the head (0 logical CPUs)
def orphan_check():
    """Run on a node after a sim actor exits: list any surviving Kit/Isaac processes."""
    out = subprocess.run(f"ps -eo pid,etimes,cmd | grep -E '{ORPHAN_PATTERN}' | grep -vE 'grep|orphan_check'",
                         shell=True, capture_output=True, text=True).stdout.strip()
    return socket.gethostname(), out.splitlines() if out else []
