"""Learner actor: central PPO + SGS merge. No Isaac import; needs only torch + rsl_rl (GPU node).

PPO decision (2026-11-19): use UW's rsl_rl fork (5.4.1, the version OmniReset trains with) unmodified.
rsl_rl's PPO separates cleanly: act()/process_env_step() fill a RolloutStorage, compute_returns()/update()
optimize whatever is in it. So workers run act()/process_env_step() locally (own policy copy) and ship their
RolloutStorage contents; the learner builds a RolloutStorage over all arrived workers' envs, then calls the
stock compute_returns() + update(). No fork of the env loop, no reimplemented PPO math.

Staleness: a worker starts rollout k+1 right after shipping k, using the weights it has. New weights are queued
on the worker actor behind that rollout, so each rollout is at most 1 update stale. PPO's ratio uses the stored
behavior log-probs, so this is the standard bounded-staleness setting.

SGS (Success-Guided Sampling, AlphaNIST idea), per reset *state*: workers keep success/attempt counters per
state in their shard and ship only the deltas with each rollout. The learner merges into global counters,
estimates p = (s + 1) / (a + 2) (Beta(1,1) posterior mean) and sets weight ∝ p(1 - p) (peaks at 50% success),
with unseen states at the max 0.25 so they still get visited. Upstream OmniReset tracks success per reset *type*
(4 buckets, 100-episode history) via SuccessMonitor.
"""
import glob
import os
import time

import ray


@ray.remote(num_gpus=1, max_restarts=-1, max_task_retries=0)
class Learner:
    def __init__(self, agent_cfg: dict, obs_dim: int, num_actions: int, envs_per_worker: int, rollout_len: int,
                 bank_ids, ckpt_dir: str, seed: int, sgs_every: int = 1, ckpt_every: int = 10,
                 init_ckpt: str | None = None, resume: bool = True):
        import copy

        import torch
        torch.manual_seed(seed)
        self.torch = torch
        self.device = "cuda:0"
        self.obs_dim, self.num_actions, self.n_env, self.L = obs_dim, num_actions, envs_per_worker, rollout_len
        self.cfg = copy.deepcopy(agent_cfg)
        self.cfg["num_steps_per_env"] = rollout_len
        self.cfg["multi_gpu"] = None
        self.sgs_every, self.ckpt_every, self.ckpt_dir = sgs_every, ckpt_every, ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        from rsl_rl.algorithms import PPO
        self.alg = PPO.construct_algorithm(self._obs_template(envs_per_worker), self._EnvSpec(envs_per_worker, num_actions),
                                           copy.deepcopy(self.cfg), self.device)
        self.alg.train_mode()
        self.version = 0
        self.history = []
        self.latest_sgs_w = None

        # SGS global counters over every bank state of the pair (sorted ids -> searchsorted lookups).
        # bank_ids: list of ids, or (bank_dir, pair) to read them from the cooked bank here (head has no pyarrow).
        if isinstance(bank_ids, (tuple, list)) and len(bank_ids) == 2 and isinstance(bank_ids[0], str):
            import pyarrow.dataset as pads
            ds = pads.dataset(os.path.join(bank_ids[0], f"pair={bank_ids[1]}"), format="parquet", partitioning="hive")
            bank_ids = ds.to_table(columns=["reset_state_id"]).column("reset_state_id").to_numpy()
        self.sgs_ids = torch.as_tensor(sorted(int(i) for i in bank_ids), device=self.device)
        self.sgs_succ = torch.zeros(len(self.sgs_ids), device=self.device)
        self.sgs_att = torch.zeros(len(self.sgs_ids), device=self.device)

        self.loaded_from = None
        latest = sorted(glob.glob(os.path.join(ckpt_dir, "learner_*.pt")))
        if resume and latest:
            self._load(latest[-1], full=True)
            self.loaded_from = latest[-1]
        elif init_ckpt:
            self.alg.load(torch.load(init_ckpt, weights_only=False, map_location=self.device),
                          {"actor": True, "critic": True}, strict=True)
            self.loaded_from = init_ckpt

    class _EnvSpec:  # the two attributes construct_algorithm reads from a VecEnv
        def __init__(self, num_envs, num_actions):
            self.num_envs, self.num_actions = num_envs, num_actions

    def _obs_template(self, n):
        from tensordict import TensorDict
        return TensorDict({"policy": self.torch.zeros(n, self.obs_dim, device=self.device)}, batch_size=[n],
                          device=self.device)

    # ---------------- weights ----------------
    def get_weights(self):
        """CPU state dicts (actor/critic incl. normalizers). Called with .remote() and the ObjectRef is passed to
        workers unopened: the head has no torch."""
        sd = self.alg.save()
        return {k: {n: t.detach().cpu() for n, t in v.items()} for k, v in sd.items()
                if k in ("actor_state_dict", "critic_state_dict")}

    def get_sgs_weights(self):
        return self.latest_sgs_w

    # ---------------- update ----------------
    def update(self, *buffers):
        """One PPO update over the rollout buffers that arrived (any subset of workers).

        Buffers are passed as top-level ObjectRef args, so Ray moves each one directly from its worker's node to
        this node (never through the driver/head)."""
        torch = self.torch
        from rsl_rl.storage import RolloutStorage
        from tensordict import TensorDict
        t0 = time.time()
        bufs = [b for b in buffers if b is not None]
        n = sum(b["obs"].shape[1] for b in bufs)
        cat = lambda k: torch.as_tensor(__import__("numpy").concatenate([b[k] for b in bufs], axis=1),  # noqa: E731
                                        device=self.device)
        obs, act, rew, done = cat("obs"), cat("actions"), cat("rewards"), cat("dones")
        val, logp, mu, sigma = cat("values"), cat("logp"), cat("mu"), cat("sigma")
        last = torch.as_tensor(__import__("numpy").concatenate([b["last_obs"] for b in bufs], axis=0), device=self.device)
        t_xfer = time.time() - t0

        t = time.time()
        storage = RolloutStorage("rl", n, self.L, self._obs_template(n), [self.num_actions], self.device)
        self.alg.storage = storage
        tr = RolloutStorage.Transition()
        for s in range(self.L):
            o = TensorDict({"policy": obs[s]}, batch_size=[n], device=self.device)
            self.alg.actor.update_normalization(o)
            self.alg.critic.update_normalization(o)
            tr.observations, tr.actions, tr.rewards, tr.dones = o, act[s], rew[s].squeeze(-1), done[s].squeeze(-1)
            tr.values, tr.actions_log_prob = val[s], logp[s].squeeze(-1)
            tr.distribution_params = (mu[s], sigma[s])
            storage.add_transition(tr)
            tr.clear()
        self.alg.compute_returns(TensorDict({"policy": last}, batch_size=[n], device=self.device))
        t_ingest = time.time() - t
        t = time.time()
        losses = self.alg.update()
        torch.cuda.synchronize()
        t_update = time.time() - t
        self.version += 1

        # SGS merge (deltas piggybacked on the buffers)
        t = time.time()
        sgs_bytes = 0
        for b in bufs:
            d = b.get("sgs")
            if d is None or len(d["ids"]) == 0:
                continue
            ids = torch.as_tensor(d["ids"], device=self.device)
            pos = torch.searchsorted(self.sgs_ids, ids)
            self.sgs_succ.index_add_(0, pos, torch.as_tensor(d["succ"], device=self.device))
            self.sgs_att.index_add_(0, pos, torch.as_tensor(d["att"], device=self.device))
            sgs_bytes += int(d["ids"].nbytes + d["succ"].nbytes + d["att"].nbytes)
        new_sgs = False
        if self.version % self.sgs_every == 0:
            p = (self.sgs_succ + 1) / (self.sgs_att + 2)
            w = p * (1 - p)
            w[self.sgs_att == 0] = 0.25
            self.latest_sgs_w = w.cpu().numpy()
            new_sgs = True
        torch.cuda.synchronize()
        t_sgs = time.time() - t

        ckpt = None
        if self.version % self.ckpt_every == 0:
            ckpt = self._save()
        att = self.sgs_att
        rec = dict(version=self.version, workers=len(bufs), envs=n, transfer_s=t_xfer, ingest_s=t_ingest,
                   update_s=t_update, sgs_merge_s=t_sgs, sgs_delta_bytes=sgs_bytes,
                   sgs_states_seen=int((att > 0).sum()), sgs_states_total=int(len(att)),
                   sgs_mean_p=float(((self.sgs_succ + 1) / (att + 2))[att > 0].mean()) if bool((att > 0).any()) else None,
                   losses={k: float(v) for k, v in losses.items()}, ckpt=ckpt, total_s=time.time() - t0)
        rec["new_sgs_weights"] = new_sgs
        self.history.append(rec)
        return rec  # python scalars only (driver-safe)

    def sgs_ids_list(self):
        return self.sgs_ids.cpu().numpy()

    def sgs_summary(self):
        a, sc = self.sgs_att, self.sgs_succ
        seen = a > 0
        p = (sc + 1) / (a + 2)
        return dict(states_total=int(len(a)), states_seen=int(seen.sum()), attempts=float(a.sum()),
                    successes=float(sc.sum()), mean_p_seen=float(p[seen].mean()) if bool(seen.any()) else None,
                    counter_bytes=int(a.numel() * a.element_size() * 2 + self.sgs_ids.numel() * 8))

    # ---------------- checkpoints ----------------
    def _save(self):
        path = os.path.join(self.ckpt_dir, f"learner_{self.version:06d}.pt")
        sd = self.alg.save()
        sd.update(iter=self.version, sgs_ids=self.sgs_ids.cpu(), sgs_succ=self.sgs_succ.cpu(),
                  sgs_att=self.sgs_att.cpu(), infos=None)
        self.torch.save(sd, path + ".tmp")
        os.replace(path + ".tmp", path)
        return path

    def _load(self, path, full):
        d = self.torch.load(path, weights_only=False, map_location=self.device)
        self.alg.load(d, None if full else {"actor": True, "critic": True}, strict=True)
        self.version = int(d.get("iter", 0))
        if "sgs_ids" in d and len(d["sgs_ids"]) == len(self.sgs_ids):
            self.sgs_succ, self.sgs_att = d["sgs_succ"].to(self.device), d["sgs_att"].to(self.device)

    def info(self):
        import socket
        return dict(hostname=socket.gethostname(), node_id=ray.get_runtime_context().get_node_id(),
                    ray_gpu_ids=[str(g) for g in ray.get_gpu_ids()],
                    device_name=self.torch.cuda.get_device_name(0), version=self.version, loaded_from=self.loaded_from)

    def ping(self):
        return self.version
