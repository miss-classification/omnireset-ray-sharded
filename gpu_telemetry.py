"""Per-actor GPU telemetry: background thread streaming `nvidia-smi dmon` (DCGM is not installed on these nodes).

Metrics (dmon -s um): sm = % of time >=1 kernel was running on the GPU (kernel-active time, NOT SM occupancy);
mem = % of time device memory was being read/written (memory-bandwidth activity); fb = framebuffer MiB used.
Samples are tagged with the current window label (e.g. "init", "step") so summaries are per stage.
Ray gives each actor exactly 1 GPU, visible as index 0 here; dmon reports physical index, so we read row 0
(1 GPU per node on this cluster; on multi-GPU nodes pass the physical index via -i).
"""
import statistics
import subprocess
import threading
import time


class GpuTelemetry:
    def __init__(self, interval_s: int = 2):
        self.interval_s = max(1, int(interval_s))
        self.samples = []  # (t, window, sm, mem, fb_mib)
        self.window = "init"
        self._stop = threading.Event()
        self._proc = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def mark(self, window: str):
        self.window = window

    def _run(self):
        cmd = ["nvidia-smi", "dmon", "-s", "um", "-d", str(self.interval_s)]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except Exception:  # noqa: BLE001
            return
        cols = None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            if line.startswith("# gpu") or line.startswith("#gpu"):
                cols = line.lstrip("#").split()
                continue
            if line.startswith("#") or not line.strip() or cols is None:
                continue
            parts = line.split()
            row = dict(zip(cols, parts))
            if row.get("gpu") not in ("0",):
                continue
            try:
                self.samples.append((time.time(), self.window, float(row["sm"]), float(row["mem"]), float(row["fb"])))
            except (KeyError, ValueError):  # '-' for unsupported fields
                continue

    @staticmethod
    def _stats(vals):
        if not vals:
            return None
        v = sorted(vals)
        p = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]  # noqa: E731
        return dict(mean=round(statistics.fmean(v), 1), p50=p(0.5), p95=p(0.95), max=v[-1], n=len(v))

    def summary(self):
        out = {}
        for w in sorted({s[1] for s in self.samples}):
            rows = [s for s in self.samples if s[1] == w]
            out[w] = dict(sm_pct=self._stats([r[2] for r in rows]), mem_bw_pct=self._stats([r[3] for r in rows]),
                          vram_mib=self._stats([r[4] for r in rows]),
                          seconds=round(rows[-1][0] - rows[0][0] + self.interval_s, 1))
        out["_meta"] = dict(source="nvidia-smi dmon -s um", interval_s=self.interval_s, samples=len(self.samples))
        return out

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
