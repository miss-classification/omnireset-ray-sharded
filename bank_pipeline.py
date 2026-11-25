"""Ray Data pipeline: cook OmniReset's released reset-state datasets into a sharded, columnar reset bank.

Why: measured in Phase 1, the dominant startup stage is torch.load of OmniReset's reset .pt files (40% of cold
init, 58% of warm; caching does not help). Those files pickle one small tensor per state per field. We cook them
once, offline and in parallel, into plain parquet with one float32 column per state dimension, plus a global
reset_state_id (needed for per-state SGS), reset type, stratum and a precomputed shard id. A SimWorker then reads
only its own shards with one bulk columnar read.

Ray Data's role here is offline generation/cooking + pre-staging shards. It does NOT accelerate in-process USD
loading or PhysX stepping.

Stages (all Ray Data, CPU tasks on GPU nodes; the head has no numpy/pyarrow and 0 CPUs):
  from_items(files) -> map_batches(load_and_flatten) -> map_batches(validate) -> filter(valid)
  -> write_parquet(partition_cols=[pair, shard])
"""
import time

RESET_TYPES = ["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped",
               "ObjectPartiallyAssembledEEGrasped"]
# OmniReset reset types -> difficulty strata (OmniReset spirit: near-goal / grasped / far-reaching)
STRATUM = {"ObjectPartiallyAssembledEEGrasped": "near_goal",
           "ObjectRestingEEGrasped": "grasped",
           "ObjectAnywhereEEGrasped": "grasped",
           "ObjectAnywhereEEAnywhere": "reaching"}
PAIRS = ["Peg__PegHole", "SquareLeg__SquareTableTop", "DrawerBottom__DrawerBox", "InsertiveCube__ReceptiveCube",
         "CupCake__Plate", "Rectangle__Wall"]
HF_BASE = ("https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/83860532010b2737aa80d6e8621235ee554186f0/"
           "Datasets/OmniReset_isaaclab3/Resets")
NUM_SHARDS = 24  # divisible by W in {1,2,3,4,6,8,12,24}: worker w of W reads shards {s : s % W == w}

# leaf paths inside dataset["initial_state"], in a fixed order -> column prefixes
LEAVES = [("articulation", "robot", "root_pose", 7), ("articulation", "robot", "root_velocity", 6),
          ("articulation", "robot", "joint_position", 12), ("articulation", "robot", "joint_velocity", 12)] + \
         [("rigid_object", o, f, n) for o in ("insertive_object", "receptive_object", "table", "ur5_metal_support")
          for f, n in (("root_pose", 7), ("root_velocity", 6))]


def column_names():
    return [f"{a}.{b}.{f}.{i}" for a, b, f, n in LEAVES for i in range(n)]


def state_id(pair_idx, type_idx, i):
    """Global, stable reset_state_id: unique across pairs/types; used as the SGS counter key."""
    return pair_idx * 10_000_000 + type_idx * 1_000_000 + i


def load_and_flatten(batch):
    """One input row = one released .pt file -> N rows (one per reset state), float32 columns."""
    import os
    import urllib.request

    import numpy as np
    import torch

    out = {k: [] for k in ("reset_state_id", "pair", "reset_type", "type_idx", "stratum", "shard")}
    cols = {c: [] for c in column_names()}
    timing = {"download_s": 0.0, "torch_load_s": 0.0, "flatten_s": 0.0}
    for pair, pair_idx, rt, type_idx in zip(batch["pair"], batch["pair_idx"], batch["reset_type"], batch["type_idx"]):
        path = f"/tmp/omnireset_{pair}_{rt}.pt"
        t = time.time()
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{HF_BASE}/{pair}/resets_{rt}.pt", path + ".part")
            os.rename(path + ".part", path)
        timing["download_s"] += time.time() - t
        t = time.time()
        d = torch.load(path, weights_only=False)
        timing["torch_load_s"] += time.time() - t
        t = time.time()
        assert d.get("quat_convention", "xyzw") == "xyzw", d.get("quat_convention")
        st = d["initial_state"]
        n = len(st["articulation"]["robot"]["joint_position"])
        k = 0
        for a, b, f, dim in LEAVES:
            arr = torch.stack(st[a][b][f]).numpy().astype(np.float32)  # (n, dim)
            assert arr.shape == (n, dim), (pair, rt, a, b, f, arr.shape)
            for i in range(dim):
                cols[column_names()[k + i]].append(arr[:, i])
            k += dim
        ids = np.array([state_id(int(pair_idx), int(type_idx), i) for i in range(n)], dtype=np.int64)
        out["reset_state_id"].append(ids)
        out["pair"].append(np.full(n, pair))
        out["reset_type"].append(np.full(n, rt))
        out["type_idx"].append(np.full(n, int(type_idx), dtype=np.int8))
        out["stratum"].append(np.full(n, STRATUM[rt]))
        out["shard"].append((ids % NUM_SHARDS).astype(np.int16))
        timing["flatten_s"] += time.time() - t
    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update({k: np.concatenate(v) for k, v in cols.items()})
    n_rows = len(res["reset_state_id"])
    for k, v in timing.items():  # per-row share of per-file stage time, summed later for stage totals
        res[f"_t_{k}"] = np.full(n_rows, v / n_rows, dtype=np.float64)
    return res


def _quat_norm(batch, prefix):
    import numpy as np
    q = np.stack([batch[f"{prefix}.{i}"] for i in range(3, 7)], axis=1)
    return np.linalg.norm(q, axis=1)


def validate(batch):
    """Basic feasibility. Flags (does not fix) bad states; `invalid_reason` names the first failed check."""
    import numpy as np
    n = len(batch["reset_state_id"])
    reason = np.full(n, "", dtype=object)

    def flag(mask, name):
        reason[(reason == "") & mask] = name

    num = [c for c in batch if c.startswith(("articulation.", "rigid_object."))]
    flag(~np.all(np.isfinite(np.stack([batch[c] for c in num], axis=1)), axis=1), "non_finite")
    for p in ("articulation.robot.root_pose", "rigid_object.insertive_object.root_pose",
              "rigid_object.receptive_object.root_pose"):
        flag(np.abs(_quat_norm(batch, p) - 1.0) > 1e-3, f"quat_norm:{p.split('.')[1]}")
    arm = np.stack([batch[f"articulation.robot.joint_position.{i}"] for i in range(6)], axis=1)
    flag(np.any(np.abs(arm) > 2 * np.pi + 1e-3, axis=1), "arm_joint_out_of_range")  # UR5e: +-2pi
    # Diagnostic only (does not invalidate): root-origin height difference. Root origins are asset-specific
    # (e.g. Rectangle__Wall's wall origin is high), so this is not a feasibility test; on 2026-11-19 it wrongly
    # dropped 39,711 Rectangle__Wall states. OmniReset generated these states with physics checks already.
    dz = batch["rigid_object.insertive_object.root_pose.2"] - batch["rigid_object.receptive_object.root_pose.2"]
    batch["diag_dz_lt_minus_5cm"] = dz < -0.05
    vel = np.stack([batch[f"rigid_object.insertive_object.root_velocity.{i}"] for i in range(6)], axis=1)
    flag(np.linalg.norm(vel, axis=1) > 5.0, "insertive_velocity_gt_5")
    batch["valid"] = reason == ""
    batch["invalid_reason"] = reason.astype(str)
    return batch


def run_pipeline(out_dir, pairs=None, ray_remote_args=None):
    """Execute the Ray Data pipeline. Returns stats dict. Call from a process that has numpy/pyarrow."""
    import ray

    pairs = pairs or PAIRS
    rra = dict(ray_remote_args or {})
    files = [dict(pair=p, pair_idx=PAIRS.index(p), reset_type=rt, type_idx=i)
             for p in pairs for i, rt in enumerate(RESET_TYPES)]
    t0 = time.time()
    ds = (ray.data.from_items(files, override_num_blocks=len(files))
          .map_batches(load_and_flatten, batch_size=1, num_cpus=2, **rra)
          .map_batches(validate, batch_format="numpy", num_cpus=1, **rra)
          .materialize())
    t_cook = time.time() - t0

    # Every operator carries ray_remote_args: Ray Data's built-in select_columns/filter/drop_columns add operators
    # that would NOT inherit the label_selector (and landed on a broken node on 2026-11-19), so the post-cook
    # steps are map_batches too.
    meta_cols = ["pair", "reset_type", "stratum", "valid", "invalid_reason", "shard", "diag_dz_lt_minus_5cm",
                 "_t_download_s", "_t_torch_load_s", "_t_flatten_s"]
    meta = ds.map_batches(lambda b: {c: b[c] for c in meta_cols}, batch_format="numpy", num_cpus=1, **rra).to_pandas()
    rows_in = len(meta)
    drop = {"valid", "invalid_reason", "diag_dz_lt_minus_5cm", "_t_download_s", "_t_torch_load_s", "_t_flatten_s"}

    def keep_valid(b):
        m = b["valid"]
        return {k: v[m] for k, v in b.items() if k not in drop}

    valid = ds.map_batches(keep_valid, batch_format="numpy", num_cpus=1, **rra)
    t1 = time.time()
    valid.write_parquet(out_dir, partition_cols=["pair", "shard"], ray_remote_args=rra)
    t_write = time.time() - t1
    wall = time.time() - t0
    rows_out = int(meta["valid"].sum())
    return dict(
        out_dir=out_dir, pairs=pairs, files=len(files), num_shards=NUM_SHARDS, columns=len(column_names()),
        rows_in=rows_in, rows_out=rows_out, rows_invalid=rows_in - rows_out,
        invalid_by_reason={k: int(v) for k, v in meta.loc[~meta["valid"], "invalid_reason"].value_counts().items()},
        rows_by_pair_type={f"{p}/{t}": int(v) for (p, t), v in meta[meta["valid"]].groupby(["pair", "reset_type"]).size().items()},
        rows_by_stratum={k: int(v) for k, v in meta[meta["valid"]]["stratum"].value_counts().items()},
        diag_dz_lt_minus_5cm_by_pair={k: int(v) for k, v in meta[meta["diag_dz_lt_minus_5cm"]]["pair"].value_counts().items()},
        stage_cpu_s=dict(download=float(meta["_t_download_s"].sum()), torch_load=float(meta["_t_torch_load_s"].sum()),
                         flatten=float(meta["_t_flatten_s"].sum())),
        cook_wall_s=t_cook, write_wall_s=t_write, pipeline_wall_s=wall, rows_per_s=rows_in / wall,
        ray_data_stats=ds.stats()[-4000:],
    )


def load_worker_shard(bank_dir, pair, worker_id, num_workers, device="cuda:0"):
    """Bulk-read this worker's shards -> dict of torch tensors in the MultiResetManager layout + ids/types.

    Plain pyarrow, no Ray Data needed at load time. Returns (state, reset_state_id, type_idx, timing).
    """
    import glob
    import os

    import numpy as np
    import pyarrow.parquet as pq
    import torch

    t0 = time.time()
    if num_workers <= NUM_SHARDS:
        shards = [s for s in range(NUM_SHARDS) if s % num_workers == worker_id]
    else:  # more simulators than shards (e.g. 4 per GPU on 8 GPUs): simulators share shards
        shards = [worker_id % NUM_SHARDS]
    files = [f for s in shards for f in glob.glob(os.path.join(bank_dir, f"pair={pair}", f"shard={s}", "*.parquet"))]
    tbl = pq.read_table(files, columns=["reset_state_id", "type_idx"] + column_names()) if files else None
    t_read = time.time() - t0
    t = time.time()
    mat = np.stack([tbl.column(c).to_numpy() for c in column_names()], axis=1)  # (n, 89) float32
    ids = tbl.column("reset_state_id").to_numpy()
    tid = tbl.column("type_idx").to_numpy().astype(np.int64)
    x = torch.from_numpy(mat).to(device)
    state, k = {}, 0
    for a, b, f, dim in LEAVES:
        state.setdefault(a, {}).setdefault(b, {})[f] = x[:, k:k + dim]
        k += dim
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return state, torch.from_numpy(ids).to(device), torch.from_numpy(tid).to(device), dict(
        files=len(files), rows=int(len(ids)), read_s=t_read, to_tensor_s=time.time() - t, total_s=time.time() - t0)
