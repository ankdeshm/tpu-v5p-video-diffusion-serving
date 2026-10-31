#!/usr/bin/env python3
"""Priority #2 & #3 (SGLang Engine): LTX-2.5 & Wan 2.2 (T2V-A14B / I2V-A14B) Serving Benchmark Suite on TPU v5p.

Runs across 6 Spot TPU v5p hosts in `ank-tpu-v5p-32-spot` (topology `2x4x4`, 4 physical v5p chips =
4 MegaCore PyTorch TPU devices / 380 GiB HBM2e / 440 GiB CPU RAM per host = 24 v5p chips across 6 hosts,
running concurrently alongside the 2 `vllm-omni` hosts on the remaining 8 v5p chips of the 32-chip slice):

Phase A — 6-Host Simultaneous Single-Host Matrix (`sglang-torchtpu-diffusion:v0.2.0`):
  - Pod 0: `Lightricks/LTX-2.5-Diffusers` — BF16, Ulysses=4 (`4` chips), XLA Fused SDPA:
      * 1-Stage (`LTX2Pipeline`, ConvVAE): `544p` (`5s` T2VA & I2VA, `10s`, `20s`) + `720p` (`5s`, `10s`) 8-step distilled
      * Two-Stage (`LTX2TwoStagePipeline`, 2x Latent Upsampler -> `1920x1088` ConvVAE):
          - `1080p Two-Stage (1920x1088) 5s 121f`
          - `1080p Kling-V4 Proxy (1920x1088) 20s 481f`
      * Dev Quality Sweep (`SGLANG_LTX25_CLEAR_DEFAULT_SIGMAS=1`): `720p (1280x704) 5s 121f (20-step, CFG=3.0)`
  - Pod 1: `Lightricks/LTX-2.5-Diffusers` — Online FP8 (`--quantization fp8`), Ulysses=4 (`4` chips), XLA Fused SDPA:
      * 1-Stage (`544p 5s/10s/20s` T2VA & I2VA + `720p 5s/10s`) + Two-Stage `1080p` (`5s 121f` & `20s 481f` Kling-V4 Proxy)
  - Pod 2: `Lightricks/LTX-2.5-Diffusers` — Parallelism, Attention & Sub-Host Chip-Scaling Ablations:
      * 2A: BF16, `CFG=2 x Ulysses=2` (`--enable-cfg-parallel`, `4` chips), XLA Fused SDPA (incl. `20-step` `CFG=3.0`)
      * 2B: BF16, Ulysses=4 (`4` chips), Unlocked TPU v5p Pallas SplashAttn (`SGLANG_TPU_V5P_SPLASH=1`)
      * 2C: BF16, `2-chip` (`num_gpus=2, ulysses_degree=2`), XLA Fused SDPA sub-host scaling
  - Pod 3: `Wan-AI/Wan2.2-I2V-A14B-Diffusers` — BF16, Ulysses=4 (`4` chips), XLA Fused SDPA across full Wan 2.2 matrix
  - Pod 4: `Wan-AI/Wan2.2-I2V-A14B-Diffusers` — Online FP8 (`Ulysses=4` full matrix + `CFG=2 x Ulysses=2` + `v5p` Pallas SplashAttn ablation)
  - Pod 5: `Wan-AI/Wan2.2-T2V-A14B-Diffusers` — BF16 (`Ulysses=4`) & Online FP8 (`Ulysses=4`) across full Wan 2.2 matrix

Phase B — Multi-Host Data-Parallel Cluster Benchmarks (`DP=6 x Ulysses=4` = 24 chips measured, plus exact `DP=8` 32-chip `v5p-32` slice sizing):
  - B1 (`LTX-2.5`): `544p 5s 121f (8-step)` at `c=6, 12` AND `1080p Kling-V4 Proxy 20s 481f` (`LTX2TwoStagePipeline`) at `c=6, 12`.
  - B2 (`Wan2.2-I2V-A14B`): `720p 5s 81f I2V (4-step fast)` at `c=6, 12` AND `720p 5s 81f I2V (20-step)` at `c=6, 12`.
  - Strictly NO cost / `$` columns.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROMPTS = [
    "A cinematic tracking shot of an ocean wave crashing against basalt sea cliffs at golden hour, realistic sea spray and Gull calls.",
    "A sleek autonomous electric concept car gliding through neon-lit Tokyo streets in light rain at night, crisp tire hiss and city ambience.",
    "Close-up wildlife footage of a red fox trotting across fresh powder snow in a sunlit pine forest, soft footsteps crunching in snow.",
    "Aerial drone shot sweeping over dramatic emerald terraced rice fields in morning mist, gentle mountain wind and distant stream.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(idx: int, msg: str) -> None:
    print(f"[{utc_now()}][sglang-pod-{idx}] {msg}", flush=True)


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def gcs_upload_bytes(bucket: str, obj: str, data: bytes, content_type: str = "application/json") -> None:
    for attempt in range(4):
        try:
            tok = vm_access_token()
            url = (
                f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
                f"?uploadType=media&name={urllib.parse.quote(obj, safe='')}"
            )
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": content_type},
            )
            with urllib.request.urlopen(req, timeout=60):
                return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def gcs_download_bytes(bucket: str, obj: str) -> bytes | None:
    tok = vm_access_token()
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def gcs_download_file(bucket: str, obj: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            tok = vm_access_token()
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(dst, "wb") as f:
                shutil.copyfileobj(resp, f, length=16 * 1024 * 1024)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def gcs_list_prefix(bucket: str, prefix: str) -> list[dict[str, Any]]:
    tok = vm_access_token()
    items: list[dict[str, Any]] = []
    pt: str | None = None
    while True:
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={urllib.parse.quote(prefix, safe='')}"
        if pt:
            url += f"&pageToken={urllib.parse.quote(pt, safe='')}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        items.extend(d.get("items", []))
        pt = d.get("nextPageToken")
        if not pt:
            break
    return items


def fast_gcs_sync_dir(idx: int, bucket: str, gcs_prefix: str, local_dir: Path) -> int:
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"
    items = gcs_list_prefix(bucket, gcs_prefix)
    blobs = [
        (it["name"][len(gcs_prefix) :], int(it["size"]))
        for it in items
        if not it["name"].endswith("/") and "/.cache/" not in it["name"]
    ]
    if not blobs:
        return 0
    local_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, Path, int, int]] = []
    chunk = 512 * 1024 * 1024
    for rel, sz in blobs:
        dst = local_dir / rel
        if dst.exists() and dst.stat().st_size == sz:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            if sz > 0:
                f.truncate(sz)
        if sz == 0:
            continue
        for s in range(0, sz, chunk):
            tasks.append((rel, dst, s, min(s + chunk, sz)))
    if not tasks:
        return len(blobs)
    t0 = time.time()
    total_b = sum(e - s for _, _, s, e in tasks)
    log(idx, f"Syncing {len(tasks)} chunks ({total_b / 1e9:.2f} GB) from gs://{bucket}/{gcs_prefix} -> {local_dir} ...")

    def dl_chunk(t: tuple[str, Path, int, int]) -> int:
        rel, dst, s, e = t
        for r in range(5):
            try:
                tok = vm_access_token()
                u = (
                    f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/"
                    f"{urllib.parse.quote(gcs_prefix + rel, safe='')}?alt=media"
                )
                req = urllib.request.Request(
                    u, headers={"Authorization": f"Bearer {tok}", "Range": f"bytes={s}-{e - 1}"}
                )
                fd = os.open(str(dst), os.O_WRONLY)
                try:
                    pos = s
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        while True:
                            buf = resp.read(16 * 1024 * 1024)
                            if not buf:
                                break
                            os.pwrite(fd, buf, pos)
                            pos += len(buf)
                finally:
                    os.close(fd)
                return e - s
            except Exception:
                if r == 4:
                    raise
                time.sleep(1.5 * (r + 1))
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(dl_chunk, tasks))
    dt = max(time.time() - t0, 1e-3)
    log(idx, f"Synced {total_b / 1e9:.2f} GB from GCS in {dt:.1f}s ({(total_b / 1e9) / dt:.2f} GB/s)")
    return len(blobs)


def apply_v5p_sglang_patches(idx: int) -> None:
    base = Path("/usr/local/lib/python3.12/site-packages")

    # 1. Unlock optional TPU v5p Pallas Splash Attention when SGLANG_TPU_V5P_SPLASH=1
    sp_path = base / "sglang_tpu/multimodal_gen/tpu_splash_attn.py"
    stxt = sp_path.read_text()
    old_chk = """        device_kind = get_pallas_tpu_device_kind()
        if device_kind not in _SUPPORTED_PALLAS_DEVICE_KINDS:
            raise RuntimeError(f"kernels are tuned for TPU v7x only, not {device_kind}")"""
    new_chk = """        import os
        device_kind = get_pallas_tpu_device_kind()
        allowed = set(_SUPPORTED_PALLAS_DEVICE_KINDS)
        if os.environ.get("SGLANG_TPU_V5P_SPLASH", "0") == "1":
            allowed.add("TPU v5p")
        if device_kind not in allowed:
            raise RuntimeError(f"kernels are tuned for TPU v7x only, not {device_kind}")"""
    if old_chk in stxt and "SGLANG_TPU_V5P_SPLASH" not in stxt:
        stxt = stxt.replace(old_chk, new_chk, 1)

    old_blk = """    mxu_dim = 256
    safe_kv_block = 2048
    vmem_budget = int(vmem_capacity_bytes * _VMEM_BUDGET_FRACTION)
    # Cap block_q: the VMEM model leaves out the double-buffered bf16 output
    # block (4 * block_q * head_dim bytes). For short KV the model alone would
    # allow block_q up to ~19.7k at block_kv=512 and overflow once that block
    # is added. 9472 is the largest size validated on device (37800x512,
    # 18900x600).
    max_block_q = 9472"""
    new_blk = """    mxu_dim = 256
    if vmem_capacity_bytes <= 32 * 2**20:
        safe_kv_block = 512
        max_block_q = 512
    else:
        safe_kv_block = 2048
        max_block_q = 9472
    vmem_budget = int(vmem_capacity_bytes * _VMEM_BUDGET_FRACTION)"""
    if old_blk in stxt:
        stxt = stxt.replace(old_blk, new_blk, 1)
    sp_path.write_text(stxt)

    # 2. Allow unpinning LTX-2.5 default_sigmas when SGLANG_LTX25_CLEAR_DEFAULT_SIGMAS=1 (for 20-step dev sweep)
    ltx_cfg = base / "sglang/multimodal_gen/configs/pipeline_configs/ltx_2_5.py"
    ltxt = ltx_cfg.read_text()
    old_sig = "default_factory=lambda: LTX25_DISTILLED_SIGMA_VALUES"
    new_sig = 'default_factory=lambda: (None if __import__("os").environ.get("SGLANG_LTX25_CLEAR_DEFAULT_SIGMAS") == "1" else LTX25_DISTILLED_SIGMA_VALUES)'
    if old_sig in ltxt and "SGLANG_LTX25_CLEAR_DEFAULT_SIGMAS" not in ltxt:
        ltx_cfg.write_text(ltxt.replace(old_sig, new_sig, 1))
    log(idx, "Applied SGLANG_TPU_V5P_SPLASH and SGLANG_LTX25_CLEAR_DEFAULT_SIGMAS hooks.")


def setup_single_host_metadata_hook(idx: int, bucket: str) -> None:
    Path("/workspace").mkdir(parents=True, exist_ok=True)
    log(idx, "Configured TPU v5p single-host isolation via TPU_SKIP_MDS_QUERY=true + TPU_WORKER_HOSTNAMES=localhost.")


def generate_reference_images(out_dir: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, str] = {}
    for size in ("960x544", "1280x704", "832x480", "1280x720", "720x1280"):
        w, h = (int(x) for x in size.split("x"))
        p = out_dir / f"ref_{size}.png"
        img = Image.new("RGB", (w, h), (30, 52, 82))
        d = ImageDraw.Draw(img)
        for y in range(0, h, 8):
            r = min(255, int(28 + 150 * (y / max(h, 1))))
            g = min(255, int(52 + 105 * (y / max(h, 1))))
            b = min(255, int(92 + 78 * (1.0 - y / max(h, 1))))
            d.rectangle([0, y, w, min(h, y + 8)], fill=(r, g, b))
        d.ellipse([w // 3, h // 4, w // 3 + 120, h // 4 + 120], fill=(248, 206, 118))
        img.save(p, format="PNG")
        res[size] = str(p)
    return res


MODEL_SPECS = {
    "ltx25": ("models/Lightricks/LTX-2.5-Diffusers/", "/dev/shm/LTX-2.5-Diffusers"),
    "wan22_i2v": ("models/Wan-AI/Wan2.2-I2V-A14B-Diffusers/", "/dev/shm/Wan2.2-I2V-A14B-Diffusers"),
    "wan22_t2v": ("models/Wan-AI/Wan2.2-T2V-A14B-Diffusers/", "/dev/shm/Wan2.2-T2V-A14B-Diffusers"),
}


def stage_single_model(idx: int, bucket: str, key: str) -> str:
    """Keep strictly ONE model checkpoint in /dev/shm at a time to leave 270+ GiB CPU RAM free."""
    gcs_prefix, local_str = MODEL_SPECS[key]
    local_dir = Path(local_str)
    for other_k, (_, other_str) in MODEL_SPECS.items():
        if other_k != key and Path(other_str).exists():
            log(idx, f"Removing {other_str} before staging {local_str}...")
            shutil.rmtree(other_str, ignore_errors=True)
    marker = local_dir / ".ready_v5p"
    if not marker.exists():
        fast_gcs_sync_dir(idx, bucket, gcs_prefix, local_dir)
        assert (local_dir / "model_index.json").exists(), f"Missing model_index.json in {local_dir}"
        marker.write_text(utc_now())
    return str(local_dir)


def alloc_free_ports(n: int) -> list[int]:
    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def kill_sglang_processes() -> None:
    for pid_s in list(os.listdir("/proc")):
        if not pid_s.isdigit() or int(pid_s) == os.getpid():
            continue
        try:
            cl = open(f"/proc/{pid_s}/cmdline", "rb").read().decode("utf-8", "ignore")
            cm = open(f"/proc/{pid_s}/comm", "rb").read().decode("utf-8", "ignore").strip()
        except OSError:
            continue
        if any(k in cl or k in cm for k in ("sglang", "sgl_diffusion", "spawn_main", "Scheduler")):
            try:
                os.kill(int(pid_s), 9)
            except OSError:
                pass
    try:
        os.remove("/tmp/libtpu_lockfile")
    except OSError:
        pass
    time.sleep(3)


def start_sglang_server(
    idx: int,
    *,
    config_name: str,
    model_path: str,
    port: int,
    num_gpus: int = 4,
    ulysses_degree: int = 4,
    cfg_parallel: bool = False,
    quantization: str | None = None,
    pipeline_class_name: str | None = None,
    v5p_splash: bool = False,
    clear_ltx_sigmas: bool = False,
    ready_timeout_s: int = 900,
) -> tuple[subprocess.Popen[Any], float, Path]:
    kill_sglang_processes()
    out_dir = Path(f"/dev/shm/sglang_out_{config_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"/workspace/sglang_{config_name}.log")

    env = os.environ.copy()
    for k in (
        "TPU_PROCESS_PORT",
        "TPU_PROCESS_ADDRESSES",
        "TPU_WORKER_ID",
        "TPU_VISIBLE_CHIPS",
        "TPU_VISIBLE_DEVICES",
        "TPU_RUNTIME_METRICS_PORTS",
        "TPU_SKIP_MDS_QUERY",
        "VBAR_CONTROL_SERVICE_URL",
        "TPU_WORKER_HOSTNAMES",
        "TPU_TOPOLOGY_WRAP",
        "TPU_TOPOLOGY_ALT",
        "LIBTPU_INIT_ARGS",
        "TPU_ML_PLATFORM",
        "TPU_PREMAPPED_BUFFER_SIZE",
        "LD_PRELOAD",
    ):
        env.pop(k, None)

    if num_gpus == 2:
        chips_bounds, topo, accel, vis_devs = "1,2,1", "1x2x1", "v5p-4", "0,1"
    else:
        chips_bounds, topo, accel, vis_devs = "2,2,1", "2x2x1", "v5p-8", "0,1,2,3"

    ports = alloc_free_ports(num_gpus + 1)
    cache_root = Path("/dev/shm/sglang_xla_cache")
    (cache_root / "xla").mkdir(parents=True, exist_ok=True)
    (cache_root / "tmp").mkdir(parents=True, exist_ok=True)

    env.update(
        {
            "TPU_SKIP_MDS_QUERY": "true",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "localhost",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(ports[0]),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{p}" for p in ports[1:]),
            "TORCH_TPU_DEVICES_PER_HOST": str(num_gpus),
            "TPU_CHIPS_PER_HOST_BOUNDS": chips_bounds,
            "CHIPS_PER_HOST_BOUNDS": chips_bounds,
            "TPU_HOST_BOUNDS": "1,1,1",
            "HOST_BOUNDS": "1,1,1",
            "TPU_TOPOLOGY": topo,
            "TOPOLOGY": topo,
            "TPU_ACCELERATOR_TYPE": accel,
            "TPU_VISIBLE_DEVICES": vis_devs,
            "JAX_PLATFORMS": "cpu",
            "PJRT_DEVICE": "TPU",
            "TPU_ML_PLATFORM": "PyTorch",
            "LIBTPU_INIT_ARGS": (
                "--xla_tpu_scoped_vmem_limit_kib=40960 "
                "--xla_tpu_enable_async_collective_fusion=true "
                "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
            ),
            "HF_HOME": "/dev/shm/hf_cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(cache_root / "xla"),
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "TMPDIR": str(cache_root / "tmp"),
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
            "SGLANG_TPU_V5P_SPLASH": "1" if v5p_splash else "0",
            "SGLANG_TPU_SPLASH_VARLEN": "1" if v5p_splash else "0",
            "SGLANG_LTX25_CLEAR_DEFAULT_SIGMAS": "1" if clear_ltx_sigmas else "0",
        }
    )

    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--num-gpus",
        str(num_gpus),
        "--ulysses-degree",
        str(ulysses_degree),
        "--attention-backend",
        "tpu_splash_attn",
        "--warmup-mode",
        "off",
        "--dit-cpu-offload=false",
        "--text-encoder-cpu-offload=false",
        "--image-encoder-cpu-offload=false",
        "--vae-cpu-offload=false",
        "--offload-during-compile=false",
        "--output-path",
        str(out_dir),
    ]
    if cfg_parallel:
        cmd.append("--enable-cfg-parallel")
    if quantization:
        cmd.extend(["--quantization", quantization])
    if pipeline_class_name:
        cmd.extend(["--pipeline-class-name", pipeline_class_name])

    log(idx, f"Starting [{config_name}]: {' '.join(cmd)} (v5p_splash={v5p_splash})")
    t0 = time.time()
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    while time.time() - t0 < ready_timeout_s:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"sglang [{config_name}] exited ({proc.returncode}):\n{tail}")
        txt = log_path.read_text(errors="replace") if log_path.exists() else ""
        if "Rank 0 scheduler is dead" in txt:
            proc.kill()
            raise RuntimeError(f"sglang [{config_name}] scheduler died:\n{txt[-4000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as r:
                if r.status == 200:
                    boot_s = round(time.time() - t0, 2)
                    log(idx, f"Server [{config_name}] READY on :{port} in {boot_s}s!")
                    return proc, boot_s, log_path
        except Exception:
            pass
        time.sleep(3)
    proc.kill()
    raise TimeoutError(f"Timed out waiting for sglang [{config_name}]")


def call_sglang_video(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 2400.0,
    proc: subprocess.Popen[Any] | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{endpoint}/v1/videos",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            job = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": e.read().decode("utf-8", "replace")[:800]}
    except Exception as exc:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(exc)[:800]}

    vid_id = job["id"]
    while time.perf_counter() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-1000:] if log_path and log_path.exists() else ""
            return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": f"rc={proc.returncode}: {tail[-600:]}"}
        if log_path is not None and log_path.exists():
            tail = log_path.read_text(errors="replace")[-3500:]
            if "Rank 0 scheduler is dead" in tail or "RESOURCE_EXHAUSTED" in tail:
                return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": tail[-800:]}
        time.sleep(0.6)
        try:
            with urllib.request.urlopen(f"{endpoint}/v1/videos/{vid_id}", timeout=30) as r:
                st = json.loads(r.read().decode("utf-8"))
        except Exception:
            continue
        if st.get("status") == "completed":
            e2e = round(time.perf_counter() - t0, 3)
            fp = st.get("file_path")
            fbytes = os.path.getsize(fp) if fp and os.path.exists(fp) else None
            if fp and os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass
            return {
                "ok": True,
                "e2e_s": e2e,
                "inference_s": st.get("inference_time_s") or e2e,
                "file_bytes": fbytes,
            }
        if st.get("status") == "failed":
            return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(st.get("error"))[:800]}
    return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": "timeout"}


def run_ltx_workloads(
    idx: int,
    *,
    bucket: str,
    gcs_prefix: str,
    pod_rows: list[dict[str, Any]],
    cfg_name: str,
    model_dir: str,
    port: int,
    precision: str,
    num_gpus: int,
    ulysses_degree: int,
    cfg_parallel: bool,
    quantization: str | None,
    pipeline_class_name: str | None,
    v5p_splash: bool,
    clear_ltx_sigmas: bool,
    workloads: list[dict[str, Any]],
    refs: dict[str, str],
) -> None:
    proc, boot_s, log_path = start_sglang_server(
        idx,
        config_name=cfg_name,
        model_path=model_dir,
        port=port,
        num_gpus=num_gpus,
        ulysses_degree=ulysses_degree,
        cfg_parallel=cfg_parallel,
        quantization=quantization,
        pipeline_class_name=pipeline_class_name,
        v5p_splash=v5p_splash,
        clear_ltx_sigmas=clear_ltx_sigmas,
    )
    ep = f"http://127.0.0.1:{port}"
    attn_lbl = "Pallas SplashAttn (v5p)" if v5p_splash else "XLA Fused SDPA"
    par_lbl = f"CFG=2xU={ulysses_degree}" if cfg_parallel else f"U={ulysses_degree}"
    try:
        for w_i, wl in enumerate(workloads):
            if proc.poll() is not None or ("Rank 0 scheduler is dead" in log_path.read_text(errors="replace")[-3000:]):
                log(idx, f"[{cfg_name}] Restarting server before {wl['label']}...")
                proc, _, log_path = start_sglang_server(
                    idx,
                    config_name=cfg_name,
                    model_path=model_dir,
                    port=port,
                    num_gpus=num_gpus,
                    ulysses_degree=ulysses_degree,
                    cfg_parallel=cfg_parallel,
                    quantization=quantization,
                    pipeline_class_name=pipeline_class_name,
                    v5p_splash=v5p_splash,
                    clear_ltx_sigmas=clear_ltx_sigmas,
                )
            p_warm: dict[str, Any] = {
                "prompt": PROMPTS[0],
                "size": wl["size"],
                "num_frames": wl["num_frames"],
                "fps": 24,
                "num_inference_steps": wl["steps"],
                "guidance_scale": wl["guidance"],
                "use_diffusion_decoder": False,
                "generator_device": "cpu",
                "seed": 100 + w_i,
            }
            if wl.get("task") == "i2va":
                p_warm["input_reference"] = refs[wl["size"]]
            log(idx, f"[{cfg_name}] Warming {wl['label']} ({wl['size']}, {wl['num_frames']}f)...")
            warm = call_sglang_video(ep, p_warm, proc=proc, log_path=log_path)
            outs = []
            wall_s = 0.0
            if warm.get("ok"):
                t0 = time.perf_counter()
                for r_i in range(wl["n"]):
                    p_r = dict(p_warm, prompt=PROMPTS[r_i % len(PROMPTS)], seed=1000 + w_i * 10 + r_i)
                    outs.append(call_sglang_video(ep, p_r, proc=proc, log_path=log_path))
                wall_s = time.perf_counter() - t0
            oks = [o for o in outs if o.get("ok")]
            med_s = round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None
            ms_step = round((med_s * 1000.0) / wl["eff_steps"], 1) if med_s else None
            rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
            row = {
                "family": "LTX-2.5",
                "config": cfg_name,
                "engine": "sglang-torchtpu",
                "pipeline": pipeline_class_name or "LTX2Pipeline",
                "task": wl.get("task", "t2va"),
                "precision": precision,
                "chips": num_gpus,
                "parallelism": par_lbl,
                "attn_mode": attn_lbl,
                "boot_s": boot_s,
                "workload": wl["label"],
                "output_res": wl["out_res"],
                "num_frames": wl["num_frames"],
                "duration_s": wl["dur"],
                "steps": wl["eff_steps"],
                "cold_compile_s": warm.get("e2e_s"),
                "warm_median_e2e_s": med_s,
                "ms_per_step": ms_step,
                "rpm": rpm,
                "video_sec_per_wall_sec": round((len(oks) * wl["dur"]) / wall_s, 3) if oks and wall_s > 0 else 0.0,
                "sample_mp4_bytes": oks[0].get("file_bytes") if oks else None,
                "ok": f"{len(oks)}/{wl['n']}",
                "error": None if oks else (warm.get("error") or (outs[0].get("error") if outs else None)),
            }
            log(idx, f"[{cfg_name}] DONE {wl['label']}: cold={warm.get('e2e_s')}s warm_med={med_s}s RPM={rpm}/min ok={row['ok']}")
            pod_rows.append(row)
            gcs_upload_bytes(bucket, f"{gcs_prefix}/pod_{idx}_results.json", json.dumps(pod_rows, indent=2).encode("utf-8"))
    finally:
        proc.terminate()
        kill_sglang_processes()


def run_wan_workloads(
    idx: int,
    *,
    bucket: str,
    gcs_prefix: str,
    pod_rows: list[dict[str, Any]],
    cfg_name: str,
    model_dir: str,
    is_i2v: bool,
    port: int,
    precision: str,
    num_gpus: int,
    ulysses_degree: int,
    cfg_parallel: bool,
    quantization: str | None,
    v5p_splash: bool,
    workloads: list[dict[str, Any]],
    refs: dict[str, str],
) -> None:
    proc, boot_s, log_path = start_sglang_server(
        idx,
        config_name=cfg_name,
        model_path=model_dir,
        port=port,
        num_gpus=num_gpus,
        ulysses_degree=ulysses_degree,
        cfg_parallel=cfg_parallel,
        quantization=quantization,
        v5p_splash=v5p_splash,
    )
    ep = f"http://127.0.0.1:{port}"
    attn_lbl = "Pallas SplashAttn (v5p)" if v5p_splash else "XLA Fused SDPA"
    par_lbl = f"CFG=2xU={ulysses_degree}" if cfg_parallel else f"U={ulysses_degree}"
    warmed_shapes: set[tuple[str, int]] = set()
    try:
        for w_i, wl in enumerate(workloads):
            if proc.poll() is not None or ("Rank 0 scheduler is dead" in log_path.read_text(errors="replace")[-3000:]):
                log(idx, f"[{cfg_name}] Restarting server before {wl['label']}...")
                proc, _, log_path = start_sglang_server(
                    idx,
                    config_name=cfg_name,
                    model_path=model_dir,
                    port=port,
                    num_gpus=num_gpus,
                    ulysses_degree=ulysses_degree,
                    cfg_parallel=cfg_parallel,
                    quantization=quantization,
                    v5p_splash=v5p_splash,
                )
                warmed_shapes.clear()

            shape_k = (wl["size"], wl["nf"])
            p_base: dict[str, Any] = {
                "prompt": PROMPTS[0],
                "size": wl["size"],
                "num_frames": wl["nf"],
                "fps": 16,
                "num_inference_steps": wl["steps"],
                "generator_device": "cpu",
                "seed": 42 + w_i,
            }
            if is_i2v:
                p_base["input_reference"] = refs[wl["size"]]

            cold_s = None
            warm_ok = True
            warm_err = None
            if shape_k not in warmed_shapes:
                p_warm = dict(p_base, num_inference_steps=min(wl["steps"], 4))
                log(idx, f"[{cfg_name}] Warming {shape_k}...")
                w_out = call_sglang_video(ep, p_warm, proc=proc, log_path=log_path)
                cold_s = w_out.get("e2e_s")
                warm_ok = bool(w_out.get("ok"))
                warm_err = w_out.get("error")
                if warm_ok:
                    warmed_shapes.add(shape_k)

            outs = []
            ws = 0.0
            if warm_ok:
                t0 = time.perf_counter()
                for r_i in range(wl["n"]):
                    pr = dict(p_base, prompt=PROMPTS[r_i % len(PROMPTS)], seed=2000 + w_i * 10 + r_i)
                    outs.append(call_sglang_video(ep, pr, proc=proc, log_path=log_path))
                ws = time.perf_counter() - t0
            oks = [o for o in outs if o.get("ok")]
            med_s = round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None
            row = {
                "family": "Wan2.2-I2V-A14B" if is_i2v else "Wan2.2-T2V-A14B",
                "config": cfg_name,
                "engine": "sglang-torchtpu",
                "task": "i2v" if is_i2v else "t2v",
                "precision": precision,
                "chips": num_gpus,
                "parallelism": par_lbl,
                "attn_mode": attn_lbl,
                "boot_s": boot_s,
                "workload": wl["label"],
                "output_res": wl["size"],
                "num_frames": wl["nf"],
                "duration_s": wl["dur"],
                "steps": wl["steps"],
                "cold_compile_s": cold_s,
                "warm_median_e2e_s": med_s,
                "ms_per_step": round((med_s * 1000.0) / wl["steps"], 1) if med_s else None,
                "rpm": round((len(oks) / ws) * 60.0, 3) if oks and ws > 0 else 0.0,
                "video_sec_per_wall_sec": round((len(oks) * wl["dur"]) / ws, 3) if oks and ws > 0 else 0.0,
                "sample_mp4_bytes": oks[0].get("file_bytes") if oks else None,
                "ok": f"{len(oks)}/{wl['n']}",
                "error": None if oks else (warm_err or (outs[0].get("error") if outs else None)),
            }
            log(idx, f"[{cfg_name}] DONE {wl['label']}: cold={cold_s}s warm_med={med_s}s RPM={row['rpm']}/min ok={row['ok']}")
            pod_rows.append(row)
            gcs_upload_bytes(bucket, f"{gcs_prefix}/pod_{idx}_results.json", json.dumps(pod_rows, indent=2).encode("utf-8"))
    finally:
        proc.terminate()
        kill_sglang_processes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-bucket", default="ank-usc1-bucket-314837540096")
    ap.add_argument("--run-id", default="v5p32_spot_sglang_serve_p23")
    ap.add_argument("--num-pods", type=int, default=6)
    args = ap.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    n_pods = args.num_pods
    gcs_prefix = f"v5p-video-serving-results/{args.run_id}"
    ip = socket.gethostbyname(socket.gethostname())
    hostname = socket.getfqdn()

    kill_sglang_processes()
    gcs_upload_bytes(
        args.gcs_bucket,
        f"{gcs_prefix}/nodes/node_{idx}.json",
        json.dumps({"idx": idx, "ip": ip, "hostname": hostname, "ts": utc_now()}).encode("utf-8"),
    )
    nodes: dict[int, dict[str, Any]] = {}
    t_w = time.time()
    while len(nodes) < n_pods and time.time() - t_w < 600:
        for i in range(n_pods):
            if i not in nodes:
                raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/nodes/node_{i}.json")
                if raw:
                    nodes[i] = json.loads(raw.decode("utf-8"))
        if len(nodes) < n_pods:
            time.sleep(3)
    log(idx, f"All {n_pods} sglang v5p-32 hosts registered!")

    apply_v5p_sglang_patches(idx)
    setup_single_host_metadata_hook(idx, args.gcs_bucket)
    refs = generate_reference_images(Path("/dev/shm/ref_inputs"))

    ltx_1stage_workloads = [
        {"label": "544p (960x544) 5s 121f T2VA (8-step distill)", "task": "t2va", "size": "960x544", "out_res": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "eff_steps": 8, "guidance": 1.0, "n": 3},
        {"label": "544p (960x544) 5s 121f I2VA (8-step distill)", "task": "i2va", "size": "960x544", "out_res": "960x544", "num_frames": 121, "dur": 5.0, "steps": 8, "eff_steps": 8, "guidance": 1.0, "n": 2},
        {"label": "544p (960x544) 10s 241f T2VA (8-step distill)", "task": "t2va", "size": "960x544", "out_res": "960x544", "num_frames": 241, "dur": 10.0, "steps": 8, "eff_steps": 8, "guidance": 1.0, "n": 2},
        {"label": "544p (960x544) 20s 481f T2VA (8-step distill)", "task": "t2va", "size": "960x544", "out_res": "960x544", "num_frames": 481, "dur": 20.0, "steps": 8, "eff_steps": 8, "guidance": 1.0, "n": 2},
        {"label": "720p (1280x704) 5s 121f T2VA (8-step distill)", "task": "t2va", "size": "1280x704", "out_res": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 8, "eff_steps": 8, "guidance": 1.0, "n": 2},
        {"label": "720p (1280x704) 10s 241f T2VA (8-step distill)", "task": "t2va", "size": "1280x704", "out_res": "1280x704", "num_frames": 241, "dur": 10.0, "steps": 8, "eff_steps": 8, "guidance": 1.0, "n": 2},
    ]

    ltx_2stage_workloads = [
        {"label": "1080p Two-Stage (1920x1088) 5s 121f (8+3 step)", "task": "t2va", "size": "1920x1088", "out_res": "1920x1088", "num_frames": 121, "dur": 5.0, "steps": 8, "eff_steps": 11, "guidance": 1.0, "n": 2},
        {"label": "1080p Kling-V4 Proxy (1920x1088) 20s 481f (8+3 step)", "task": "t2va", "size": "1920x1088", "out_res": "1920x1088", "num_frames": 481, "dur": 20.0, "steps": 8, "eff_steps": 11, "guidance": 1.0, "n": 2},
    ]

    ltx_dev20_workloads = [
        {"label": "720p (1280x704) 5s 121f (20-step dev CFG=3.0)", "task": "t2va", "size": "1280x704", "out_res": "1280x704", "num_frames": 121, "dur": 5.0, "steps": 20, "eff_steps": 20, "guidance": 3.0, "n": 2},
    ]

    wan_workloads = [
        {"label": "480p (832x480) 5s 81f (4-step fast)", "size": "832x480", "nf": 81, "dur": 5.0, "steps": 4, "n": 3},
        {"label": "480p (832x480) 5s 81f (20-step)", "size": "832x480", "nf": 81, "dur": 5.0, "steps": 20, "n": 2},
        {"label": "480p (832x480) 5s 81f (40-step full)", "size": "832x480", "nf": 81, "dur": 5.0, "steps": 40, "n": 2},
        {"label": "720p (1280x720) 5s 81f (4-step fast)", "size": "1280x720", "nf": 81, "dur": 5.0, "steps": 4, "n": 2},
        {"label": "720p (1280x720) 5s 81f (20-step)", "size": "1280x720", "nf": 81, "dur": 5.0, "steps": 20, "n": 2},
        {"label": "720p (1280x720) 5s 81f (40-step full)", "size": "1280x720", "nf": 81, "dur": 5.0, "steps": 40, "n": 2},
        {"label": "720p (1280x720) 10s 161f (20-step)", "size": "1280x720", "nf": 161, "dur": 10.0, "steps": 20, "n": 2},
        {"label": "720p Portrait (720x1280) 5s 81f (20-step)", "size": "720x1280", "nf": 81, "dur": 5.0, "steps": 20, "n": 2},
    ]

    port = 30110 + idx
    pod_rows: list[dict[str, Any]] = []

    # Phase A: 6-Host Simultaneous Single-Host Matrix
    if idx == 0:
        ltx_dir = stage_single_model(idx, args.gcs_bucket, "ltx25")
        run_ltx_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="ltx25_sglang_bf16_u4_1stage", model_dir=ltx_dir, port=port,
            precision="BF16", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
            quantization=None, pipeline_class_name=None, v5p_splash=False, clear_ltx_sigmas=False,
            workloads=ltx_1stage_workloads, refs=refs,
        )
        run_ltx_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="ltx25_sglang_bf16_u4_2stage_1080p", model_dir=ltx_dir, port=port,
            precision="BF16", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
            quantization=None, pipeline_class_name="LTX2TwoStagePipeline", v5p_splash=False, clear_ltx_sigmas=False,
            workloads=ltx_2stage_workloads, refs=refs,
        )
        run_ltx_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="ltx25_sglang_bf16_u4_dev20", model_dir=ltx_dir, port=port,
            precision="BF16", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
            quantization=None, pipeline_class_name=None, v5p_splash=False, clear_ltx_sigmas=True,
            workloads=ltx_dev20_workloads, refs=refs,
        )

    elif idx == 1:
        ltx_dir = stage_single_model(idx, args.gcs_bucket, "ltx25")
        run_ltx_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="ltx25_sglang_fp8_u4_1stage", model_dir=ltx_dir, port=port,
            precision="FP8 (online)", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
            quantization="fp8", pipeline_class_name=None, v5p_splash=False, clear_ltx_sigmas=False,
            workloads=ltx_1stage_workloads, refs=refs,
        )
        run_ltx_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="ltx25_sglang_fp8_u4_2stage_1080p", model_dir=ltx_dir, port=port,
            precision="FP8 (online)", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
            quantization="fp8", pipeline_class_name="LTX2TwoStagePipeline", v5p_splash=False, clear_ltx_sigmas=False,
            workloads=ltx_2stage_workloads, refs=refs,
        )

    elif idx == 2:
        ltx_dir = stage_single_model(idx, args.gcs_bucket, "ltx25")
        ablation_ltx = [
            ltx_1stage_workloads[0],  # 544p 5s
            ltx_1stage_workloads[2],  # 544p 10s
            ltx_1stage_workloads[4],  # 720p 5s
        ]
        # 2A: CFG=2 x Ulysses=2 (including 20-step CFG=3.0)
        run_ltx_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="ltx25_sglang_bf16_cfg2_u2", model_dir=ltx_dir, port=port,
            precision="BF16", num_gpus=4, ulysses_degree=2, cfg_parallel=True,
            quantization=None, pipeline_class_name=None, v5p_splash=False, clear_ltx_sigmas=True,
            workloads=ablation_ltx + ltx_dev20_workloads, refs=refs,
        )
        # 2B: Unlocked v5p Pallas SplashAttention vs SDPA
        try:
            run_ltx_workloads(
                idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
                cfg_name="ltx25_sglang_bf16_u4_pallas_v5p", model_dir=ltx_dir, port=port,
                precision="BF16", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
                quantization=None, pipeline_class_name=None, v5p_splash=True, clear_ltx_sigmas=False,
                workloads=ablation_ltx, refs=refs,
            )
        except Exception as exc:
            log(idx, f"LTX Pallas v5p note: {exc}")
        # 2C: 2-chip (`num_gpus=2, ulysses_degree=2`) sub-host scaling
        try:
            run_ltx_workloads(
                idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
                cfg_name="ltx25_sglang_bf16_2chip_u2", model_dir=ltx_dir, port=port,
                precision="BF16", num_gpus=2, ulysses_degree=2, cfg_parallel=False,
                quantization=None, pipeline_class_name=None, v5p_splash=False, clear_ltx_sigmas=False,
                workloads=ablation_ltx, refs=refs,
            )
        except Exception as exc:
            log(idx, f"LTX 2-chip note: {exc}")

    elif idx == 3:
        i2v_dir = stage_single_model(idx, args.gcs_bucket, "wan22_i2v")
        run_wan_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="wan22_i2v_sglang_bf16_u4", model_dir=i2v_dir, is_i2v=True, port=port,
            precision="BF16", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
            quantization=None, v5p_splash=False, workloads=wan_workloads, refs=refs,
        )

    elif idx == 4:
        i2v_dir = stage_single_model(idx, args.gcs_bucket, "wan22_i2v")
        run_wan_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="wan22_i2v_sglang_fp8_u4", model_dir=i2v_dir, is_i2v=True, port=port,
            precision="FP8 (online)", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
            quantization="fp8", v5p_splash=False, workloads=wan_workloads, refs=refs,
        )
        wan_short = [wan_workloads[0], wan_workloads[3], wan_workloads[4]]  # 480p 4-step, 720p 4-step, 720p 20-step
        run_wan_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="wan22_i2v_sglang_fp8_cfg2_u2", model_dir=i2v_dir, is_i2v=True, port=port,
            precision="FP8 (online)", num_gpus=4, ulysses_degree=2, cfg_parallel=True,
            quantization="fp8", v5p_splash=False, workloads=wan_short, refs=refs,
        )
        try:
            run_wan_workloads(
                idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
                cfg_name="wan22_i2v_sglang_fp8_u4_pallas_v5p", model_dir=i2v_dir, is_i2v=True, port=port,
                precision="FP8 (online)", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
                quantization="fp8", v5p_splash=True, workloads=wan_short, refs=refs,
            )
        except Exception as exc:
            log(idx, f"Wan Pallas v5p note: {exc}")

    elif idx == 5:
        t2v_dir = stage_single_model(idx, args.gcs_bucket, "wan22_t2v")
        run_wan_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="wan22_t2v_sglang_bf16_u4", model_dir=t2v_dir, is_i2v=False, port=port,
            precision="BF16", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
            quantization=None, v5p_splash=False, workloads=wan_workloads, refs=refs,
        )
        run_wan_workloads(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            cfg_name="wan22_t2v_sglang_fp8_u4", model_dir=t2v_dir, is_i2v=False, port=port,
            precision="FP8 (online)", num_gpus=4, ulysses_degree=4, cfg_parallel=False,
            quantization="fp8", v5p_splash=False, workloads=wan_workloads, refs=refs,
        )

    # Synchronize all 6 pods before Phase B multi-host cluster benchmarks
    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phaseA_pod_{idx}.txt", b"done\n", content_type="text/plain")
    log(idx, f"Phase A complete! Waiting for all {n_pods} sglang pods...")
    t_s = time.time()
    while time.time() - t_s < 7200:
        if sum(1 for i in range(n_pods) if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phaseA_pod_{i}.txt") is not None) == n_pods:
            break
        time.sleep(10)

    cluster_rows: list[dict[str, Any]] = []

    # Phase B1: 6-Host DP=6 (24 chips) LTX-2.5 Cluster (1-Stage 544p 5s + Two-Stage 1080p 20s Kling-V4 Proxy!)
    ltx_dir = stage_single_model(idx, args.gcs_bucket, "ltx25")
    for b1_tag, pipe_cls, b1_suites in [
        (
            "ltx1s",
            None,
            [
                ("LTX-2.5 544p 5s 121f (8-step)", "960x544", 121, 5.0, 8, n_pods, n_pods * 2),
                ("LTX-2.5 544p 5s 121f (8-step)", "960x544", 121, 5.0, 8, n_pods * 2, n_pods * 3),
            ],
        ),
        (
            "ltx2s",
            "LTX2TwoStagePipeline",
            [
                ("LTX-2.5 1080p Kling-V4 Proxy 20s 481f (Two-Stage)", "1920x1088", 481, 20.0, 8, n_pods, n_pods),
                ("LTX-2.5 1080p Kling-V4 Proxy 20s 481f (Two-Stage)", "1920x1088", 481, 20.0, 8, n_pods * 2, n_pods * 2),
            ],
        ),
    ]:
        dp_port = 30210 + idx
        proc_dp, _, log_dp = start_sglang_server(
            idx, config_name=f"dp_{b1_tag}_{idx}", model_path=ltx_dir, port=dp_port,
            num_gpus=4, ulysses_degree=4, pipeline_class_name=pipe_cls,
        )
        try:
            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_{b1_tag}_{idx}.txt", f"http://{ip}:{dp_port}".encode("utf-8"), content_type="text/plain")
            if idx == 0:
                eps: dict[int, str] = {}
                t_e = time.time()
                while len(eps) < n_pods and time.time() - t_e < 1200:
                    for i in range(n_pods):
                        b = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_{b1_tag}_{i}.txt")
                        if b:
                            eps[i] = b.decode("utf-8").strip()
                    if len(eps) < n_pods:
                        time.sleep(3)
                for lbl, sz, nf, dur, steps, conc, n_req in b1_suites:
                    log(0, f"[Cluster DP={n_pods}] Warming all {n_pods} hosts on {lbl}...")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=n_pods) as p:
                        w_outs = list(p.map(lambda i: call_sglang_video(eps[i], {"prompt": PROMPTS[0], "size": sz, "num_frames": nf, "fps": 24, "num_inference_steps": steps, "guidance_scale": 1.0, "use_diffusion_decoder": False, "generator_device": "cpu", "seed": 42}), range(n_pods)))
                    if not any(w.get("ok") for w in w_outs):
                        cluster_rows.append({"family": "LTX-2.5", "workload": lbl, "dp_replicas": n_pods, "chips": n_pods * 4, "concurrency": conc, "ok": f"0/{n_req}", "error": w_outs[0].get("error")})
                        continue
                    t_c = time.perf_counter()
                    reqs = [(eps[r % n_pods], {"prompt": PROMPTS[r % 4], "size": sz, "num_frames": nf, "fps": 24, "num_inference_steps": steps, "guidance_scale": 1.0, "use_diffusion_decoder": False, "generator_device": "cpu", "seed": 900 + r}) for r in range(n_req)]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as p:
                        outs = list(p.map(lambda it: call_sglang_video(it[0], it[1]), reqs))
                    ws = time.perf_counter() - t_c
                    oks = [o for o in outs if o.get("ok")]
                    rpm_dp6 = round((len(oks) / ws) * 60.0, 3) if oks and ws > 0 else 0.0
                    rpm_dp8_slice = round(rpm_dp6 * (8.0 / n_pods), 3)
                    cluster_rows.append(
                        {
                            "family": "LTX-2.5",
                            "engine": "sglang-torchtpu",
                            "pipeline": pipe_cls or "LTX2Pipeline",
                            "workload": lbl,
                            "dp_replicas_measured": n_pods,
                            "chips_measured": n_pods * 4,
                            "concurrency": conc,
                            "wall_s": round(ws, 3),
                            "median_e2e_s": round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None,
                            "measured_rpm": rpm_dp6,
                            "v5p32_dp8_slice_rpm": rpm_dp8_slice,
                            "v5p32_slices_for_30_rpm": math.ceil(30.0 / rpm_dp8_slice) if rpm_dp8_slice > 0 else None,
                            "video_sec_per_wall_sec_measured": round((len(oks) * dur) / ws, 3) if oks and ws > 0 else 0.0,
                            "video_sec_per_wall_sec_v5p32": round(((len(oks) * dur) / ws) * (8.0 / n_pods), 3) if oks and ws > 0 else 0.0,
                            "ok": f"{len(oks)}/{n_req}",
                        }
                    )
                    log(0, f"[Cluster DP={n_pods}] DONE {lbl} c={conc}: RPM(DP={n_pods})={rpm_dp6} -> v5p-32(DP=8)={rpm_dp8_slice} vids/min")
                gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_{b1_tag}.txt", b"ok\n", content_type="text/plain")
            else:
                t_h = time.time()
                while time.time() - t_h < 3600:
                    if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/done_{b1_tag}.txt") is not None:
                        break
                    time.sleep(5)
        finally:
            proc_dp.terminate()
            kill_sglang_processes()

    # Phase B2: 6-Host DP=6 (24 chips) Wan2.2-I2V-A14B Cluster
    i2v_dir = stage_single_model(idx, args.gcs_bucket, "wan22_i2v")
    dp_port = 30310 + idx
    proc_w, _, _ = start_sglang_server(
        idx, config_name=f"dp_wan_{idx}", model_path=i2v_dir, port=dp_port,
        num_gpus=4, ulysses_degree=4, quantization="fp8",
    )
    try:
        gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_wan_{idx}.txt", f"http://{ip}:{dp_port}".encode("utf-8"), content_type="text/plain")
        if idx == 0:
            eps = {}
            t_e = time.time()
            while len(eps) < n_pods and time.time() - t_e < 1200:
                for i in range(n_pods):
                    b = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_wan_{i}.txt")
                    if b:
                        eps[i] = b.decode("utf-8").strip()
                if len(eps) < n_pods:
                    time.sleep(3)
            for lbl, sz, nf, dur, steps, conc, n_req in [
                ("Wan2.2-I2V-A14B 720p 5s 81f (4-step fast, FP8)", "1280x720", 81, 5.0, 4, n_pods, n_pods * 2),
                ("Wan2.2-I2V-A14B 720p 5s 81f (4-step fast, FP8)", "1280x720", 81, 5.0, 4, n_pods * 2, n_pods * 3),
                ("Wan2.2-I2V-A14B 720p 5s 81f (20-step, FP8)", "1280x720", 81, 5.0, 20, n_pods, n_pods),
                ("Wan2.2-I2V-A14B 720p 5s 81f (20-step, FP8)", "1280x720", 81, 5.0, 20, n_pods * 2, n_pods * 2),
            ]:
                log(0, f"[Cluster DP={n_pods}] Warming all {n_pods} hosts on {lbl}...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_pods) as p:
                    list(p.map(lambda i: call_sglang_video(eps[i], {"prompt": PROMPTS[0], "size": sz, "num_frames": nf, "fps": 16, "num_inference_steps": steps, "input_reference": refs[sz], "generator_device": "cpu", "seed": 42}), range(n_pods)))
                t_c = time.perf_counter()
                reqs = [(eps[r % n_pods], {"prompt": PROMPTS[r % 4], "size": sz, "num_frames": nf, "fps": 16, "num_inference_steps": steps, "input_reference": refs[sz], "generator_device": "cpu", "seed": 3000 + r}) for r in range(n_req)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as p:
                    outs = list(p.map(lambda it: call_sglang_video(it[0], it[1]), reqs))
                ws = time.perf_counter() - t_c
                oks = [o for o in outs if o.get("ok")]
                rpm_dp6 = round((len(oks) / ws) * 60.0, 3) if oks and ws > 0 else 0.0
                rpm_dp8_slice = round(rpm_dp6 * (8.0 / n_pods), 3)
                cluster_rows.append(
                    {
                        "family": "Wan2.2-I2V-A14B",
                        "engine": "sglang-torchtpu",
                        "precision": "FP8 (online)",
                        "workload": lbl,
                        "dp_replicas_measured": n_pods,
                        "chips_measured": n_pods * 4,
                        "concurrency": conc,
                        "wall_s": round(ws, 3),
                        "median_e2e_s": round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None,
                        "measured_rpm": rpm_dp6,
                        "v5p32_dp8_slice_rpm": rpm_dp8_slice,
                        "v5p32_slices_for_30_rpm": math.ceil(30.0 / rpm_dp8_slice) if rpm_dp8_slice > 0 else None,
                        "video_sec_per_wall_sec_measured": round((len(oks) * dur) / ws, 3) if oks and ws > 0 else 0.0,
                        "video_sec_per_wall_sec_v5p32": round(((len(oks) * dur) / ws) * (8.0 / n_pods), 3) if oks and ws > 0 else 0.0,
                        "ok": f"{len(oks)}/{n_req}",
                    }
                )
                log(0, f"[Cluster DP={n_pods}] DONE {lbl} c={conc}: RPM(DP={n_pods})={rpm_dp6} -> v5p-32(DP=8)={rpm_dp8_slice} vids/min")

            all_host = []
            for i in range(n_pods):
                r = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{i}_results.json")
                if r:
                    all_host.extend(json.loads(r.decode("utf-8")))
            gcs_upload_bytes(
                args.gcs_bucket,
                f"{gcs_prefix}/summary.json",
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "updated_utc": utc_now(),
                        "nodes": nodes,
                        "single_host_results": all_host,
                        "cluster_results": cluster_rows,
                    },
                    indent=2,
                ).encode("utf-8"),
            )
            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt", b"ok\n", content_type="text/plain")
            log(0, "Priority #2 & #3 (SGLang Engine) TPU v5p-32 benchmark suite 100% COMPLETE!")
        else:
            t_h = time.time()
            while time.time() - t_h < 3600:
                if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt") is not None:
                    break
                time.sleep(5)
    finally:
        proc_w.terminate()
        kill_sglang_processes()


if __name__ == "__main__":
    main()
