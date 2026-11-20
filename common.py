"""Driver-side helpers shared by every phase: run metadata, cost, json output. Ray + stdlib only (head has no numpy)."""
import json
import os
import shutil
import time

import ray

import config


class RunMeta:
    """Wraps a phase run: records wall time, node usage, seeds, and a $ estimate into the phase json."""

    def __init__(self, phase: str):
        self.phase = phase
        self.t0 = time.time()
        self.seeds = {}

    def note_seed(self, worker_id: int, seed: int):
        self.seeds[str(worker_id)] = seed

    def finish(self, gpu_nodes_used: int):
        wall = time.time() - self.t0
        alive_gpu = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
        hrs = wall / 3600
        return dict(
            phase=self.phase,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.t0)),
            wall_s=round(wall, 1),
            gpu_node_type=config.GPU_NODE_TYPE, gpu_nodes_used=gpu_nodes_used,
            gpu_nodes_alive_in_cluster=len(alive_gpu),
            head_node_type=config.HEAD_NODE_TYPE,
            usd_per_hr=dict(gpu_node=config.GPU_NODE_USD_PER_HR, head=config.HEAD_NODE_USD_PER_HR),
            # cost of the nodes this phase used, and of the whole cluster for the same wall time
            # (idle GPU nodes are billed too)
            usd_used_nodes=round(hrs * (gpu_nodes_used * config.GPU_NODE_USD_PER_HR + config.HEAD_NODE_USD_PER_HR), 3),
            usd_whole_cluster=round(hrs * (len(alive_gpu) * config.GPU_NODE_USD_PER_HR + config.HEAD_NODE_USD_PER_HR), 3),
            pricing_note="GCP on-demand list price approx., us-central1; excludes Anyscale fees, storage, egress",
            base_seed=config.SEED, worker_seeds=self.seeds,
            excluded_node_prefixes=config.EXCLUDE_NODE_ID_PREFIXES,
            hardware_note="NVIDIA A100-SXM4-80GB on GCP; config is cloud-portable",
        )


def write_json(name: str, obj: dict):
    """Write to cluster storage and mirror into the workspace results/ folder."""
    path = os.path.join(config.RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.makedirs(config.LOCAL_RESULTS_DIR, exist_ok=True)
    shutil.copy(path, os.path.join(config.LOCAL_RESULTS_DIR, name))
    print(f"wrote {path} (+ results/{name})")
    return path
