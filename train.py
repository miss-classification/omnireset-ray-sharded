"""Train PPO with many Isaac Sim simulators: --gpus GPUs x --sims-per-gpu simulators x --envs-per-sim envs, plus one
GPU for the learner. Recommended (80 GB GPUs): --sims-per-gpu 4 --envs-per-sim 16384 --patch physics_only_nofabric_scaled.

  python train.py --gpus 7 --sims-per-gpu 4 --envs-per-sim 16384 --iterations 200     # 8 GPUs in total

All simulators start at the same time; startup depends on envs per simulator, not on the total. With
--sims-per-gpu > 1 the simulators on a GPU share it through NVIDIA MPS (started and stopped by this script).
Output: <RESULTS_DIR>/train_<tag>.json, checkpoints in <CKPT_DIR>/<tag>/.
"""
import argparse

import ray

import config
import mps
import train_loop
from common import RunMeta, write_json

SAMPLES_PER_MINIBATCH = 655_360   # PPO minibatch size used up to ~1M envs in our runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, required=True, help="GPUs for simulators (the learner takes one more)")
    ap.add_argument("--sims-per-gpu", type=int, default=4, help="simulators per GPU (MPS when > 1)")
    ap.add_argument("--envs-per-sim", type=int, default=16384)
    ap.add_argument("--iterations", type=int, default=100, help="PPO updates (each = 32 steps from every simulator)")
    ap.add_argument("--patch", default="physics_only_nofabric_scaled", help="scene_patches function; 'none' to disable")
    ap.add_argument("--mini-batches", type=int, default=None,
                    help=f"PPO minibatches per epoch (default: about {SAMPLES_PER_MINIBATCH:,} samples each)")
    ap.add_argument("--resume", default=None, help="learner checkpoint to start from")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    W = args.gpus * args.sims_per_gpu
    total = W * args.envs_per_sim
    samples = total * config.ROLLOUT_LEN
    mb = args.mini_batches or max(4, round(samples / SAMPLES_PER_MINIBATCH))
    tag = args.tag or f"train_{args.gpus}gpu_{args.sims_per_gpu}x{args.envs_per_sim}"
    print(f"{args.gpus} GPUs x {args.sims_per_gpu} simulators x {args.envs_per_sim:,} envs = {total:,} envs "
          f"(+1 learner GPU) | {samples:,} samples per update, {mb} minibatches | patch {args.patch}")

    ray.init(address=config.RAY_ADDRESS)
    meta = RunMeta(tag)
    try:
        res = train_loop.run(tag, W, args.envs_per_sim, args.iterations, meta=meta, sims_per_gpu=args.sims_per_gpu,
                             cfg_patch=None if args.patch == "none" else args.patch, init_ckpt=args.resume,
                             agent_overrides={"algorithm.num_mini_batches": mb})
    finally:   # MPS daemons are stopped by train_loop on success; make sure on failure too
        if args.sims_per_gpu > 1:
            gpu_nodes = [n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
            mps.on_nodes("stop", gpu_nodes)
    its = res["iterations"][2:] or res["iterations"]
    eps = sum(r["agg_env_steps_per_s"] for r in its) / max(1, len(its))
    sr = [r["success"] / r["attempts"] for r in res["iterations"] if r.get("attempts")]
    print(f"\n{total:,} envs | startup {res['init_wall_s'] / 60:.1f} min (all simulators in parallel) | "
          f"{eps:,.0f} env-steps/s | success {sr[0]:.3f} -> {sr[-1]:.3f}" if sr else "")
    out = dict(meta=meta.finish(gpu_nodes_used=args.gpus + 1), args=vars(args), minibatches=mb,
               steady_env_steps_per_s=eps, result=res)
    write_json(f"{tag}.json", out)


if __name__ == "__main__":
    main()
