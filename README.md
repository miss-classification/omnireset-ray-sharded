# isaac-ray-scaling

Isaac Lab simulation sharded across GPUs with Ray: one Isaac Sim process per GPU (a Ray actor), one PPO learner
actor, reset states served from a Ray Data bank, per-reset-state success-guided sampling, and automatic restart of
failed simulators. Tested with the OmniReset peg-insertion task (UR5e + Robotiq 2F-85) on A100-80GB GPUs.

## Results (A100-80GB, PhysX TGS)

| Setup | Startup | Training speed |
|---|---:|---:|
| 1 GPU, 1 simulator x 16,384 envs | 8.9 min | 18.7K env-steps/s |
| 1 GPU, 4 simulators x 4,096 with NVIDIA MPS | 3.6 min | 19.3K env-steps/s |
| 4 GPUs, 4 simulators x 4,096 | 2.5 min | |
| 5 GPUs, 5 x 16,384 = 81,920 envs | 9.4 min | 101K env-steps/s (96% of linear) |

## Stack

Isaac Sim 6.1.0.0, Isaac Lab 3.0 (ae37b02), UWLab 2.0 (2b40afb, OmniReset), UW rsl_rl 5.4.1, torch 2.11 cu128,
Ray 2.60, Python 3.12. Exact Python packages: `image/lockfile.txt`.

## Files

| File | Purpose |
|---|---|
| `config.py` | all settings: task, envs per GPU, number of workers, paths, cost rates |
| `sim_worker.py` | `SimWorker` Ray actor: boots Isaac Sim on its GPU, builds the env, runs rollouts, startup timers, GPU telemetry |
| `learner.py` | `Learner` Ray actor: PPO update on rollouts from all workers, merges success-guided sampling counters, checkpoints |
| `train_loop.py` | driver loop: starts workers in parallel, gathers rollouts with timeouts, rejoins restarted workers |
| `bank_pipeline.py` | Ray Data pipeline that converts OmniReset reset states into per-worker parquet shards |
| `scene_patches.py` | optional scene changes, e.g. one shared physics material per object |
| `gpu_telemetry.py`, `common.py` | GPU sampling (`nvidia-smi dmon`), result files, run cost |
| `cluster_env.py` | installs the stack on a node at runtime when the image does not have it (no-op on the image below) |
| `image/` | Containerfile for an Anyscale image with the full stack (`make_containerfile.py` embeds the lockfile) |

## Run

```
python check_gpus.py                     # one GPU task per node, placement check
python build_reset_bank.py               # Ray Data: reset states -> per-worker shards
python scale_training.py --smoke         # 2 workers x 1,024 envs, 3 iterations
python scale_training.py --ws 1 2 4      # scaling at config.ENVS_PER_GPU envs per worker
python fault_tolerance.py                # kill one worker mid-run, watch it restart and rejoin
python learner_placement.py              # learner on its own GPU vs sharing a worker's GPU
python one_gpu.py                        # one GPU: several simulators with MPS, and next-run setup during training
python startup_and_memory.py             # host RAM and physics materials per env
python memory_sweep.py                   # GPU memory and throughput vs envs per GPU
python teardown.py                       # stop actors, check for leftover Isaac processes
```

Image: `cd image && python make_containerfile.py && anyscale image build -n isaac-ray-scaling -f Containerfile.isaac`

## Notes

- Isaac Sim never runs on the driver; each actor uses `cuda:0` of the GPU Ray assigns to it.
- One PhysX process allows 65,536 physics materials. The OmniReset peg task copies 2 per env, so one simulator holds
  at most 32,767 envs; more simulators raise the total.
- Host RAM: about 5.5 GB per simulator plus 2.1 MB per env.
- MPS: `one_gpu.py` starts the MPS daemon on the node with a Ray task and stops it afterwards.
- Source of the task, assets and policies: [OmniReset](https://arxiv.org/abs/2603.15789),
  [UWLab](https://github.com/UW-Lab/UWLab), [uwlab-assets](https://huggingface.co/datasets/UW-Lab/uwlab-assets).
