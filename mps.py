"""NVIDIA MPS on Ray nodes: several simulators share one GPU and run their kernels at the same time.

Without MPS, processes on one GPU take turns (time slicing): 4 simulators on one GPU started 2.5x faster but stepped
at 0.45x. With MPS they step at full speed. A Ray task starts the MPS control daemon on each node (as the ray user,
pipe dir in /tmp); every actor on that node that should use MPS gets MPS_ENV. Stop the daemon when the run ends.
"""
import time

import ray

MPS_ENV = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps-pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-mps-log"}


@ray.remote(num_cpus=0)
def mps(cmd):
    """cmd: 'start' or 'stop'. Runs on the node it is scheduled to (use label_selector={"ray.io/node-id": id})."""
    import os
    import subprocess
    env = dict(os.environ, **MPS_ENV)
    for d in MPS_ENV.values():
        os.makedirs(d, exist_ok=True)
    if cmd == "start":
        r = subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, capture_output=True, text=True)
    else:
        r = subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", env=env, capture_output=True, text=True)
    time.sleep(2)
    ps = subprocess.run("pgrep -a nvidia-cuda-mps", shell=True, capture_output=True, text=True).stdout
    return dict(rc=r.returncode, out=(r.stdout + r.stderr)[-300:], procs=ps.strip())


def on_nodes(cmd, node_ids, timeout=120):
    """Start or stop MPS on every node in node_ids; returns {node_id: result}."""
    refs = {n: mps.options(label_selector={"ray.io/node-id": n}).remote(cmd) for n in set(node_ids)}
    return {n: ray.get(r, timeout=timeout) for n, r in refs.items()}
