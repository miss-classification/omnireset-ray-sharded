"""Single config for every phase. Scaling to more nodes/GPUs is a change here only."""
import os

# ---- cluster ----
RAY_ADDRESS = "auto"
NUM_GPUS_EXPECTED = 8          # GPUs Phase 0 expects to find
NUM_WORKERS = 6                # W: SimWorker actors (1 GPU each). 6 while 2 nodes are broken (see EXCLUDE_NODE_IDS)

# Nodes Ray must not schedule onto (exclusion, not pinning), e.g. nodes whose Ray workers cannot start.
EXCLUDE_NODE_ID_PREFIXES = []  # node-id prefixes to keep work off (e.g. unhealthy nodes)

# ---- paths ----
STORAGE = "/mnt/cluster_storage"
RESULTS_DIR = os.path.join(STORAGE, "demo_results")
CKPT_DIR = os.path.join(STORAGE, "demo_ckpts")
BANK_DIR = os.path.join(RESULTS_DIR, "reset_bank")
KIT_CACHE_DIR = "/tmp/kit_cache"  # node-local; was /mnt/cluster_storage/kit_cache (shared NFS) before 2026-11-19 23:00

# ---- Isaac process env (not set in the workspace image; passed to every sim actor) ----
ISAAC_ENV_VARS = {
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "ACCEPT_EULA": "Y",
    "PRIVACY_CONSENT": "Y",
    "OMNI_KIT_ALLOW_ROOT": "1",
    # per-node, node-local kit/shader cache (not shared NFS): each node pays its own first compile once
    "OMNI_KIT_CACHEDIR": KIT_CACHE_DIR,
    "TERM": "xterm",
}

# ---- task: OmniReset (UWLab 2.0), UR5e + Robotiq 2F-85, peg -> peg hole ----
TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"
TASK_OVERRIDES = ["env.scene.insertive_object=peg", "env.scene.receptive_object=peghole"]
AGENT_ENTRY = "rsl_rl_cfg_entry_point"

# ---- sim / training ----
NUM_ENVS_PHASE1 = 1024
STEPS_PHASE1 = 500
ROLLOUT_LEN = 32               # L
SEED = 0                       # base seed; worker w uses worker_seed(w) = SEED + w


def worker_seed(worker_id: int) -> int:
    return SEED + worker_id

BANK_SIZE = None               # None = all released OmniReset states (Peg__PegHole: 40,350). No synthetic states.
BANK_PAIR = "Peg__PegHole"     # reset-bank partition matching TASK_OVERRIDES
SGS_EVERY = 1                  # learner recomputes + broadcasts SGS weights every K updates
CKPT_EVERY = 10                # learner checkpoint every K updates -> CKPT_DIR/<run tag>/
ENVS_PER_GPU = 16384           # Phase 2: last good point below PhysX's 64K-material crash at 32,768 (VRAM 23%)
PAPER_ENVS_PER_GPU = 4000      # AlphaNIST reference density
OMNIRESET_TOTAL_ENVS = 65536   # OmniReset scales PPO to 64K+ envs

# ---- cost (list prices, on-demand, us-central1; verify against current GCP pricing) ----
GPU_NODE_TYPE = "a2-ultragpu-1g"   # 1x A100-SXM4-80GB, 12 vCPU, 170 GB
GPU_NODE_USD_PER_HR = 5.07         # GCP list price a2-ultragpu-1g us-central1 (approx.)
HEAD_NODE_TYPE = "n2-standard-16"
HEAD_NODE_USD_PER_HR = 0.78        # GCP list price n2-standard-16 us-central1 (approx.)
# Excludes Anyscale platform fees, storage and egress.

# ---- telemetry ----
TELEMETRY_INTERVAL_S = 2           # nvidia-smi dmon sample period inside each sim actor

# ---- local mirror of results (workspace-visible) ----
LOCAL_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def actor_label_selector(nodes):
    """Ray label_selector excluding EXCLUDE_NODE_ID_PREFIXES. `nodes` = ray.nodes()."""
    bad = [n["NodeID"] for n in nodes if any(n["NodeID"].startswith(p) for p in EXCLUDE_NODE_ID_PREFIXES)]
    return {"ray.io/node-id": f"!in({','.join(bad)})"} if bad else None
