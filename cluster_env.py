"""Provision a GPU node with UWLab's pinned OmniReset stack, idempotently.

Stack (pinned by UWLab 2.0, the OmniReset codebase, github.com/UW-Lab/UWLab @ 2b40afb):
  Isaac Sim 6.1.0.0 | Isaac Lab @ ae37b02 (3.0.0-EA, editable) | UW rsl_rl fork 5.4.1
  torch 2.11.0+cu128 | uwlab, uwlab_tasks, uwlab_rl, uwlab_assets (editable) | Python 3.12

Why not Isaac Sim 4.5 / Isaac Lab 2.1: the workspace image is Python 3.12 and Ray
needs one Python minor cluster-wide. 4.5 wheels are cp310, 5.x cp311. UWLab 2.0 itself targets
6.1 / py3.12, so this is the *faithful* OmniReset stack, not a compromise.

Source is staged once on shared storage (/mnt/cluster_storage/uwlab_src/UWLab, both repos at
the pins above); each node copies it to local disk and runs UWLab's own installer (uwlab.sh -i).
Head node is never provisioned. A replacement node self-provisions on first actor start.
"""
import fcntl
import os
import subprocess
import sys
import time

SHARED_SRC = "/mnt/cluster_storage/uwlab_src/UWLab"
LOCAL_SRC = "/home/ray/UWLab"
PIN = "uwlab-2b40afb+isaaclab-ae37b02+isaacsim-6.1.0.0"
MARKER = "/home/ray/.uwlab_stack"
LOCK = "/tmp/uwlab_install.lock"
UV = os.path.expanduser("~/.local/bin/uv")
VENV = os.path.dirname(os.path.dirname(sys.executable))  # /home/ray/venv


def is_ready():
    try:
        return open(MARKER).read().strip() == PIN
    except OSError:
        return False


def _run(cmd, env=None, log=print, timeout=5400):
    log("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    r = subprocess.run(cmd, shell=isinstance(cmd, str), env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}\n{r.stdout[-3000:]}\n{r.stderr[-5000:]}")
    return r.stdout


def ensure_stack(log=print):
    """Returns dict(installed_now, seconds)."""
    t0 = time.time()
    if is_ready():
        return dict(installed_now=False, seconds=0.0)
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if not is_ready():
                env = dict(os.environ, TERM="xterm", VIRTUAL_ENV=VENV, PATH=f"{VENV}/bin:{os.path.dirname(UV)}:" + os.environ["PATH"])
                # 1. node-local copy of the pinned source (editable installs must not live on NFS)
                _run(f"rm -rf {LOCAL_SRC} && cp -a {SHARED_SRC} {LOCAL_SRC}", log=log)
                # 2. drop the Isaac Lab 3.0.0 wheel from the earlier provisioning pass, if present
                subprocess.run([UV, "pip", "uninstall", "--python", sys.executable, "isaaclab"],
                               capture_output=True, text=True)
                # 3. Isaac Sim 6.1.0.0 (UWLab pip_installation step)
                _run([UV, "pip", "install", "--python", sys.executable, "isaacsim[all,extscache]==6.1.0.0",
                      "--extra-index-url", "https://pypi.nvidia.com", "--index-strategy", "unsafe-best-match"],
                     env=env, log=log)
                # 4. UWLab's own installer: Isaac Lab @ pin (editable), torch cu128, uwlab_* (editable), UW rsl_rl
                _run(f"cd {LOCAL_SRC} && ./uwlab.sh --install rsl_rl", env=env, log=log)
                # 5. converge on lockfile.txt (uv pip freeze of a healthy node; same file the image installs).
                #    Pins every transitive dep, incl. urllib3, which Ray workers import at startup
                #    (ray -> requests -> urllib3); an interrupted urllib3 swap left 2 nodes unable to start workers.
                here = os.path.dirname(os.path.abspath(__file__))
                lock = next(p for p in (os.path.join(here, "lockfile.txt"), os.path.join(here, "image", "lockfile.txt"))
                            if os.path.exists(p))
                # --no-deps: the lock is a full freeze; re-resolving it fails (google-api-core 2.44 vs requests 2.32.3
                # as frozen on the original node).
                _run([UV, "pip", "install", "--python", sys.executable, "--no-deps", "-r", lock,
                      "--extra-index-url", "https://pypi.nvidia.com",
                      "--extra-index-url", "https://download.pytorch.org/whl/cu128",
                      "--index-strategy", "unsafe-best-match"], env=env, log=log)
                with open(MARKER, "w") as f:
                    f.write(PIN)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return dict(installed_now=True, seconds=time.time() - t0)
