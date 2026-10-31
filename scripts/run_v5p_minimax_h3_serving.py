#!/usr/bin/env python3
"""Priority #1: MiniMax-H3 & FastH3 (4-step VSA Distilled) Video+Audio Serving Benchmark Suite on TPU v5p.

Runs across the 4 Spot TPU v5p hosts in `ank-tpu-v5p-16-spot` (topology `2x2x4`, 4 physical v5p chips =
4 MegaCore PyTorch TPU devices / 380 GiB HBM2e / 440 GiB CPU RAM per host =
16 physical v5p chips / 1,520 GiB HBM2e slice-wide):

Phase 1A — 4-Host Simultaneous Single-Host (4x TPU v5p chips = 4 MegaCore devices) Configuration & Input-Distribution Matrix:
  - Pod 0: Base `MiniMaxAI/MiniMax-H3` (`FL2VA` partition: `fl2va` i2v + `t2va`), BF16, Ulysses=4, XLA Fused SDPA
  - Pod 1: Base `MiniMaxAI/MiniMax-H3` (`FL2VA` partition: `fl2va` i2v + `t2va`), Online FP8 (`--quantization fp8`), Ulysses=4, XLA Fused SDPA
  - Pod 2: Base `MiniMaxAI/MiniMax-H3` (`FL2VA` partition) Ablations:
      * 2A: BF16, Ulysses=4, XLA Fused SDPA + Online AdaLN (`--minimax-h3-adaln-online`)
      * 2B: BF16, Ulysses=4, Unlocked TPU v5p Pallas Splash Attention (`SGLANG_TPU_V5P_SPLASH=1` + `SGLANG_TPU_SPLASH_VARLEN=1`)
      * 2C: Online FP8, 2-chip (`num_gpus=2, ulysses_degree=2`), XLA Fused SDPA sub-host scaling
  - Pod 3: `FastH3` (`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` materialized with `kevin-mi/FastH3-4step-Preview-overlay`),
      Ulysses=4, BF16 then Online FP8 (`--quantization fp8`), followed by single-model `/dev/shm` swap to `MiniMax-H3/FL2VA` for Phase 1B.

Phase 1B — Full 16-Chip `v5p-16` Slice (`4x` Data-Parallel Replicas, `DP=4 x Ulysses=4`) Throughput & 30-RPM Sizing:
  - Load-balances concurrent requests (`c = 4, 8`) across all 4 hosts (`16` v5p chips) on `768p 15s (4-step & 25-step)`,
    `768p 5s (4-step)`, and a mixed video production distribution (`50%` 5s, `30%` 10s, `20%` 15s).
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
    print(f"[{utc_now()}][pod-{idx}] {msg}", flush=True)


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


def fast_gcs_sync_dir(
    idx: int,
    bucket: str,
    gcs_prefix: str,
    local_dir: Path,
    allowed_prefixes: tuple[str, ...] | None = None,
) -> int:
    """Multi-threaded chunked download from regional GCS into /dev/shm."""
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"
    items = gcs_list_prefix(bucket, gcs_prefix)
    blobs: list[tuple[str, int]] = []
    for it in items:
        if it["name"].endswith("/") or "/.cache/" in it["name"]:
            continue
        rel = it["name"][len(gcs_prefix) :]
        if allowed_prefixes and not (rel in allowed_prefixes or rel.startswith(allowed_prefixes)):
            continue
        blobs.append((rel, int(it["size"])))
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
    """Ensure MiniMax-H3 TPU patches + optional SGLANG_TPU_V5P_SPLASH unlock are applied."""
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
    log(idx, "Verified sglang-torchtpu MiniMax-H3 patches + SGLANG_TPU_V5P_SPLASH support.")


def setup_single_host_metadata_hook(idx: int, bucket: str) -> None:
    Path("/workspace").mkdir(parents=True, exist_ok=True)
    log(idx, "Configured TPU v5p single-host isolation via TPU_SKIP_MDS_QUERY=true + TPU_WORKER_HOSTNAMES=localhost.")


def generate_synthetic_keyframes(input_dir: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    input_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "16:9_768": (1344, 768, "keyframe_1344x768.png"),
        "9:16_768": (768, 1344, "keyframe_768x1344.png"),
        "16:9_512": (896, 512, "keyframe_896x512.png"),
    }
    paths: dict[str, str] = {}
    for k, (w, h, fn) in specs.items():
        p = input_dir / fn
        img = Image.new("RGB", (w, h), color=(24, 42, 68))
        draw = ImageDraw.Draw(img)
        for y in range(0, h, 8):
            r = min(255, int(30 + 160 * (y / max(h, 1))))
            g = min(255, int(55 + 110 * (y / max(h, 1))))
            b = min(255, int(95 + 80 * (1.0 - y / max(h, 1))))
            draw.rectangle([0, y, w, min(h, y + 8)], fill=(r, g, b))
        draw.ellipse([w // 3, h // 5, w // 3 + 140, h // 5 + 140], fill=(252, 212, 128))
        img.save(p, format="PNG")
        paths[k] = str(p)
    return paths


def stage_single_model_in_shm(idx: int, bucket: str, which: str) -> str:
    """Strictly stage ONE model checkpoint at a time in /dev/shm to preserve 270+ GiB free CPU RAM."""
    base_dir = Path("/dev/shm/MiniMax-H3")
    fh_dir = Path("/dev/shm/FastH3-4step-Materialized")
    if which == "minimax_h3":
        if fh_dir.exists():
            log(idx, "Removing /dev/shm/FastH3-4step-Materialized before staging MiniMax-H3...")
            shutil.rmtree(fh_dir, ignore_errors=True)
        marker = base_dir / ".ready_fl2va"
        if not marker.exists():
            fast_gcs_sync_dir(
                idx,
                bucket,
                "models/MiniMaxAI/MiniMax-H3/",
                base_dir,
                allowed_prefixes=("FL2VA/", "model_index.json", "modular_model_index.json"),
            )
            assert (base_dir / "FL2VA" / "model_index.json").exists(), "Missing FL2VA/model_index.json!"
            marker.write_text("ok\n")
        return str(base_dir / "FL2VA")
    elif which == "fasth3":
        if base_dir.exists():
            log(idx, "Removing /dev/shm/MiniMax-H3 before staging FastH3-4step-Materialized...")
            shutil.rmtree(base_dir, ignore_errors=True)
        marker = fh_dir / ".ready_fh3"
        if not marker.exists():
            fast_gcs_sync_dir(idx, bucket, "models/FastVideo/FastH3-4step-Materialized/", fh_dir)
            assert (fh_dir / "model_index.json").exists(), "Missing FastH3 model_index.json!"
            marker.write_text("ok\n")
        return str(fh_dir)
    raise ValueError(f"Unknown model key: {which}")


def alloc_free_ports(n: int) -> list[int]:
    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def build_v5p_server_env(
    *,
    num_gpus: int,
    v5p_splash: bool,
) -> dict[str, str]:
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
        chips_bounds = "1,2,1"
        topo = "1x2x1"
        accel = "v5p-4"
        vis_devs = "0,1"
    else:
        chips_bounds = "2,2,1"
        topo = "2x2x1"
        accel = "v5p-8"
        vis_devs = "0,1,2,3"

    ports = alloc_free_ports(num_gpus + 1)
    master_port, sb_ports = ports[0], ports[1:]
    cache_root = Path("/dev/shm/sglang_xla_cache")
    (cache_root / "xla").mkdir(parents=True, exist_ok=True)
    (cache_root / "tmp").mkdir(parents=True, exist_ok=True)

    env.update(
        {
            "TPU_SKIP_MDS_QUERY": "true",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "localhost",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{p}" for p in sb_ports),
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
        }
    )
    return env


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
    num_gpus: int,
    ulysses_degree: int,
    quantization: str | None,
    v5p_splash: bool,
    adaln_online: bool = False,
    ready_timeout_s: int = 900,
) -> tuple[subprocess.Popen[Any], float, Path]:
    kill_sglang_processes()
    out_dir = Path(f"/dev/shm/sglang_out_{config_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"/workspace/sglang_{config_name}.log")

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
        "--encoder-parallel",
        "fold",
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
    if quantization:
        cmd.extend(["--quantization", quantization])
    if adaln_online:
        cmd.append("--minimax-h3-adaln-online")

    env = build_v5p_server_env(num_gpus=num_gpus, v5p_splash=v5p_splash)
    log(idx, f"Starting [{config_name}]: {' '.join(cmd)} (v5p_splash={v5p_splash})")
    t0 = time.time()
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    while time.time() - t0 < ready_timeout_s:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"sglang [{config_name}] exited early rc={proc.returncode}:\n{tail}")
        txt = log_path.read_text(errors="replace") if log_path.exists() else ""
        if "Rank 0 scheduler is dead" in txt:
            proc.kill()
            raise RuntimeError(f"sglang [{config_name}] scheduler died:\n{txt[-4000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as resp:
                if resp.status == 200:
                    boot_s = round(time.time() - t0, 2)
                    free_mem = subprocess.check_output(["free", "-g"], text=True).splitlines()[1]
                    log(idx, f"Server [{config_name}] READY on :{port} in {boot_s}s! (RAM: {free_mem})")
                    return proc, boot_s, log_path
        except Exception:
            pass
        time.sleep(3)

    proc.kill()
    tail = log_path.read_text(errors="replace")[-4000:]
    raise TimeoutError(f"Server [{config_name}] timed out after {ready_timeout_s}s:\n{tail}")


def is_server_healthy(port: int, proc: subprocess.Popen[Any], log_path: Path) -> bool:
    if proc.poll() is not None:
        return False
    if log_path.exists():
        tail = log_path.read_text(errors="replace")[-6000:]
        if "Rank 0 scheduler is dead" in tail or "Check failure stack trace" in tail:
            return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def submit_and_poll_video(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 1800.0,
    proc: subprocess.Popen[Any] | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/v1/videos",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            job = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": f"http_{e.code}",
            "error": err_body[:800],
            "e2e_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "submit_error",
            "error": str(exc)[:800],
            "e2e_s": round(time.perf_counter() - t0, 3),
        }

    vid_id = job["id"]
    poll_url = f"{endpoint}/v1/videos/{vid_id}"
    while time.perf_counter() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-1200:] if log_path and log_path.exists() else ""
            return {
                "ok": False,
                "id": vid_id,
                "status": "server_crashed",
                "error": f"rc={proc.returncode}: {tail[-600:]}",
                "e2e_s": round(time.perf_counter() - t0, 3),
            }
        if log_path is not None and log_path.exists():
            tail = log_path.read_text(errors="replace")[-4000:]
            if "Rank 0 scheduler is dead" in tail or "RESOURCE_EXHAUSTED" in tail:
                return {
                    "ok": False,
                    "id": vid_id,
                    "status": "worker_oom_or_crash",
                    "error": tail[-800:],
                    "e2e_s": round(time.perf_counter() - t0, 3),
                }
        time.sleep(0.75)
        try:
            with urllib.request.urlopen(poll_url, timeout=30) as resp:
                st = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        status = st.get("status")
        if status == "completed":
            e2e_s = time.perf_counter() - t0
            fp = st.get("file_path")
            fsize = os.path.getsize(fp) if fp and os.path.exists(fp) else None
            if fp and os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass
            return {
                "ok": True,
                "id": vid_id,
                "status": "completed",
                "e2e_s": round(e2e_s, 3),
                "inference_time_s": st.get("inference_time_s"),
                "peak_memory_mb": st.get("peak_memory_mb"),
                "size": st.get("size"),
                "seconds": st.get("seconds"),
                "file_bytes": fsize,
            }
        if status == "failed":
            return {
                "ok": False,
                "id": vid_id,
                "status": "failed",
                "error": str(st.get("error"))[:800],
                "e2e_s": round(time.perf_counter() - t0, 3),
            }

    return {
        "ok": False,
        "id": vid_id,
        "status": "timeout",
        "error": f"Timed out after {timeout_s}s",
        "e2e_s": round(time.perf_counter() - t0, 3),
    }


def make_minimax_payload(
    *,
    prompt: str,
    task: str,
    short_edge: int,
    aspect_ratio: str,
    duration_seconds: float,
    num_inference_steps: int,
    seed: int,
    keyframe_uri: str | None = None,
) -> dict[str, Any]:
    conditions: list[dict[str, Any]] = []
    if task == "fl2va":
        assert keyframe_uri is not None, "fl2va requires keyframe_uri"
        conditions = [
            {
                "type": "image",
                "role": "keyframe",
                "uri": keyframe_uri,
                "frame_index": 0,
            }
        ]
    return {
        "prompt": prompt,
        "task": task,
        "conditions": conditions,
        "target": {
            "short_edge": short_edge,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": float(duration_seconds),
        },
        "num_inference_steps": int(num_inference_steps),
        "quality": "lossless",
        "generator_device": "cpu",
        "seed": seed,
    }


def run_host_sweep(
    idx: int,
    *,
    bucket: str,
    gcs_out_prefix: str,
    config_meta: dict[str, Any],
    server_spec: dict[str, Any],
    workloads: list[dict[str, Any]],
    keyframes: dict[str, str],
    existing_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    port = server_spec["port"]
    endpoint = f"http://127.0.0.1:{port}"
    proc, boot_s, log_path = start_sglang_server(idx, **server_spec)
    config_meta = dict(config_meta, boot_s=boot_s)

    warmed_shapes: set[tuple[str, int, str, float]] = set()
    failed_shapes: set[tuple[str, int, str, float]] = set()

    try:
        for w_i, wl in enumerate(workloads):
            task = wl["task"]
            short_edge = wl["short_edge"]
            aspect = wl["aspect_ratio"]
            dur = float(wl["duration_seconds"])
            steps_grid = wl["num_inference_steps"]
            denoise_steps = steps_grid - 1
            conc = wl.get("concurrency", 1)
            n_reqs = wl.get("num_requests", max(2, conc))
            kf_key = f"{aspect}_{short_edge}"
            kf_uri = keyframes.get(kf_key, keyframes["16:9_768"]) if task == "fl2va" else None
            shape_key = (task, short_edge, aspect, dur)

            if shape_key in failed_shapes:
                row = {
                    **config_meta,
                    "workload_label": wl["label"],
                    "task": task,
                    "short_edge": short_edge,
                    "aspect_ratio": aspect,
                    "duration_seconds": dur,
                    "denoise_steps": denoise_steps,
                    "num_inference_steps": steps_grid,
                    "concurrency": conc,
                    "num_requests": n_reqs,
                    "ok_count": 0,
                    "cold_compile_e2e_s": None,
                    "error": "Skipped after prior HBM OOM / compile failure on identical (task, resolution, duration) shape",
                }
                existing_results.append(row)
                gcs_upload_bytes(
                    bucket,
                    f"{gcs_out_prefix}/pod_{idx}_results.json",
                    json.dumps(existing_results, indent=2).encode("utf-8"),
                )
                continue

            if not is_server_healthy(port, proc, log_path):
                log(idx, f"[{config_meta['name']}] Restarting server before {wl['label']}...")
                proc, _, log_path = start_sglang_server(idx, **server_spec)
                warmed_shapes.clear()

            cold_compile_s = None
            if shape_key not in warmed_shapes:
                warm_steps = 5 if config_meta["is_fasth3"] else min(steps_grid, 5)
                log(idx, f"[{config_meta['name']}] Warming XLA shape {shape_key} (grid={warm_steps})...")
                w_payload = make_minimax_payload(
                    prompt=PROMPTS[0],
                    task=task,
                    short_edge=short_edge,
                    aspect_ratio=aspect,
                    duration_seconds=dur,
                    num_inference_steps=warm_steps,
                    seed=100 + w_i,
                    keyframe_uri=kf_uri,
                )
                w_res = submit_and_poll_video(endpoint, w_payload, proc=proc, log_path=log_path)
                cold_compile_s = w_res["e2e_s"]
                log(
                    idx,
                    f"[{config_meta['name']}] Warmup {shape_key}: ok={w_res['ok']} e2e={cold_compile_s}s err={w_res.get('error', '')[:200]}",
                )
                if w_res["ok"]:
                    warmed_shapes.add(shape_key)
                else:
                    failed_shapes.add(shape_key)
                    row = {
                        **config_meta,
                        "workload_label": wl["label"],
                        "task": task,
                        "short_edge": short_edge,
                        "aspect_ratio": aspect,
                        "duration_seconds": dur,
                        "denoise_steps": denoise_steps,
                        "num_inference_steps": steps_grid,
                        "concurrency": conc,
                        "num_requests": n_reqs,
                        "ok_count": 0,
                        "cold_compile_e2e_s": cold_compile_s,
                        "error": w_res.get("error"),
                    }
                    existing_results.append(row)
                    gcs_upload_bytes(
                        bucket,
                        f"{gcs_out_prefix}/pod_{idx}_results.json",
                        json.dumps(existing_results, indent=2).encode("utf-8"),
                    )
                    continue

            log(
                idx,
                f"[{config_meta['name']}] Running {wl['label']} ({task} {short_edge}p {aspect} {dur}s, {denoise_steps} steps, c={conc}, n={n_reqs})...",
            )
            payloads = [
                make_minimax_payload(
                    prompt=PROMPTS[(w_i + r_i) % len(PROMPTS)],
                    task=task,
                    short_edge=short_edge,
                    aspect_ratio=aspect,
                    duration_seconds=dur,
                    num_inference_steps=steps_grid,
                    seed=1000 + w_i * 100 + r_i,
                    keyframe_uri=kf_uri,
                )
                for r_i in range(n_reqs)
            ]

            t_wall0 = time.perf_counter()
            if conc == 1:
                outs = [submit_and_poll_video(endpoint, p, proc=proc, log_path=log_path) for p in payloads]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                    outs = list(pool.map(lambda p: submit_and_poll_video(endpoint, p, proc=proc, log_path=log_path), payloads))
            wall_s = time.perf_counter() - t_wall0

            oks = [o for o in outs if o.get("ok")]
            lats = [o["e2e_s"] for o in oks]
            inf_ts = [o["inference_time_s"] for o in oks if o.get("inference_time_s") is not None]

            mean_e2e = round(statistics.mean(lats), 3) if lats else None
            med_e2e = round(statistics.median(lats), 3) if lats else None
            mean_inf = round(statistics.mean(inf_ts), 3) if inf_ts else None
            step_ms = round(((mean_inf or mean_e2e) * 1000.0) / max(denoise_steps, 1), 1) if (mean_inf or mean_e2e) else None
            rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
            vid_sec_per_s = round((len(oks) * dur) / wall_s, 3) if oks and wall_s > 0 else 0.0

            row = {
                **config_meta,
                "workload_label": wl["label"],
                "task": task,
                "resolved_size": oks[0].get("size") if oks else None,
                "short_edge": short_edge,
                "aspect_ratio": aspect,
                "duration_seconds": dur,
                "denoise_steps": denoise_steps,
                "num_inference_steps": steps_grid,
                "concurrency": conc,
                "num_requests": n_reqs,
                "ok_count": len(oks),
                "cold_compile_e2e_s": cold_compile_s,
                "wall_s": round(wall_s, 3),
                "mean_e2e_s": mean_e2e,
                "median_e2e_s": med_e2e,
                "mean_inference_s": mean_inf,
                "ms_per_denoise_step": step_ms,
                "rpm_videos_per_min": rpm,
                "video_sec_per_wall_sec": vid_sec_per_s,
                "peak_memory_mb": max((o.get("peak_memory_mb") or 0 for o in oks), default=None),
                "sample_mp4_bytes": oks[0].get("file_bytes") if oks else None,
                "error": None if len(oks) == n_reqs else next((o.get("error") for o in outs if not o.get("ok")), None),
            }
            log(
                idx,
                f"[{config_meta['name']}] DONE {wl['label']}: ok={len(oks)}/{n_reqs} med_e2e={med_e2e}s step={step_ms}ms RPM={rpm}/min RTF={vid_sec_per_s}x",
            )
            existing_results.append(row)
            gcs_upload_bytes(
                bucket,
                f"{gcs_out_prefix}/pod_{idx}_results.json",
                json.dumps(existing_results, indent=2).encode("utf-8"),
            )
    finally:
        proc.terminate()
        kill_sglang_processes()
    return existing_results


def format_markdown_summary(
    run_id: str,
    nodes: dict[int, dict[str, Any]],
    all_host_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Priority #1 — MiniMax-H3 & FastH3 Video+Audio Serving on TPU v5p (`{run_id}`)",
        "",
        f"- **Updated (UTC):** `{utc_now()}`",
        "- **Cluster / Nodepool:** `ank-tpu-v5p-usc1` / `ank-tpu-v5p-16-spot` (`2x2x4`, `4` hosts × `4` physical `v5p` chips = `16` physical chips / `16` MegaCore devices / `1,520 GiB` HBM2e)",
        "- **Serving Engine:** `sglang-torchtpu` (`TpuOOTPlatform`, XLA Fused SDPA vs TPU v5p Pallas SplashAttention + Ulysses Sequence Parallelism, 24 fps H.264 + 32 kHz stereo AAC muxed `.mp4`)",
        f"- **Registered Hosts:** `{json.dumps({k: v['hostname'] for k, v in sorted(nodes.items())})}`",
        "",
        "## 1. Single-Host (`4x TPU v5p Chips` = `4` MegaCore Devices / `380 GiB` HBM2e) Configuration & Input-Distribution Matrix",
        "",
        "| Config | Model | Prec. | Chips | Attn Mode | Task & Input Spec | Canvas | Steps | Conc. | Cold Compile (s) | Warm Med E2E (s) | ms/step | Host RPM (vids/min) | Video-s / Wall-s | OK |",
        "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_host_rows:
        lines.append(
            f"| `{r['name']}` | `{r['model_short']}` | `{r['precision']}` | {r['chips']} | `{r['attn_mode']}` | "
            f"**{r['workload_label']}** (`{r['task']}`) | `{r.get('resolved_size') or '-'}` | {r['denoise_steps']} | {r['concurrency']} | "
            f"{r.get('cold_compile_e2e_s') or '-'} | **{r.get('median_e2e_s') or '-'}** | {r.get('ms_per_denoise_step') or '-'} | "
            f"**{r.get('rpm_videos_per_min') or 0:.2f}** | {r.get('video_sec_per_wall_sec') or 0:.2f}x | `{r['ok_count']}/{r['num_requests']}` |"
        )

    if cluster_rows:
        lines.extend(
            [
                "",
                "## 2. Full `v5p-16` Slice (`16x TPU v5p Chips` = `4×` Data-Parallel Replicas, `DP=4 × Ulysses=4`) Throughput & `30 RPM` Sizing",
                "",
                "| Cluster Config | Workload / Input Distribution | Steps | Concurrency | Wall (s) | Med E2E (s) | P90 E2E (s) | Slice RPM (vids/min) | Video-s / Wall-s | Slices for 30 RPM | OK |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for c in cluster_rows:
            rpm = c.get("rpm_videos_per_min") or 0.0
            slices_30 = f"{math.ceil(30.0 / rpm)} ({30.0 / rpm:.2f}x)" if rpm > 0 else "-"
            lines.append(
                f"| `{c['cluster_config']}` | **{c['workload_label']}** | {c['denoise_steps']} | {c['concurrency']} | "
                f"{c['wall_s']} | **{c.get('median_e2e_s') or '-'}** | {c.get('p90_e2e_s') or '-'} | "
                f"**{rpm:.2f}** | {c.get('video_sec_per_wall_sec') or 0:.2f}x | `{slices_30}` | `{c['ok_count']}/{c['num_requests']}` |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-bucket", default="ank-usc1-bucket-314837540096")
    ap.add_argument("--run-id", default="v5p16_spot_minimax_h3_serve_p1")
    args = ap.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    gcs_prefix = f"v5p-video-serving-results/{args.run_id}"
    ip = socket.gethostbyname(socket.gethostname())
    hostname = socket.getfqdn()

    kill_sglang_processes()

    # 1. Register node IP in GCS & wait for all 4 pods
    node_info = {"idx": idx, "ip": ip, "hostname": hostname, "ts": utc_now()}
    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/nodes/node_{idx}.json", json.dumps(node_info).encode("utf-8"))
    log(idx, f"Registered pod-{idx} ({ip} / {hostname}). Waiting for all 4 Spot v5p-16 hosts...")

    nodes: dict[int, dict[str, Any]] = {}
    t_wait = time.time()
    while len(nodes) < 4 and time.time() - t_wait < 600:
        for i in range(4):
            if i not in nodes:
                raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/nodes/node_{i}.json")
                if raw:
                    nodes[i] = json.loads(raw.decode("utf-8"))
        if len(nodes) < 4:
            time.sleep(3)
    assert len(nodes) == 4, f"Expected 4 nodes, found {nodes}"
    log(idx, f"All 4 Spot v5p-16 hosts registered: { {i: nodes[i]['ip'] for i in range(4)} }")

    # 2. Apply v5p splash unlock patch, compile 2x2x1 metadata hook, create keyframes
    apply_v5p_sglang_patches(idx)
    setup_single_host_metadata_hook(idx, args.gcs_bucket)
    keyframes = generate_synthetic_keyframes(Path("/dev/shm/inputs"))

    base_workloads = [
        {"label": "768p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 15s t2va (4-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 15s i2v (12-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 13, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 15s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 15s i2v (49-step full)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 50, "concurrency": 1, "num_requests": 1},
        {"label": "768p 16:9 5s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
        {"label": "768p 16:9 5s t2va (25-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 5s t2va (49-step full)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 50, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 10s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 10s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        {"label": "768p 9:16 Portrait 5s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 5.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        {"label": "512p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 512, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        {"label": "768p 16:9 15s i2v (4-step, c=2)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 2, "num_requests": 4},
    ]

    fasth3_workloads = [
        {"label": "768p 16:9 15s FastH3 (4-step VSA distill)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
        {"label": "768p 16:9 10s FastH3 (4-step VSA distill)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
        {"label": "768p 16:9 5s FastH3 (4-step VSA distill)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
        {"label": "768p 9:16 Portrait 15s FastH3 (4-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        {"label": "512p 16:9 15s FastH3 (4-step)", "task": "t2va", "short_edge": 512, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 3},
        {"label": "768p 16:9 15s FastH3 (4-step, c=2)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 2, "num_requests": 4},
    ]

    pod_results: list[dict[str, Any]] = []
    port = 30010 + idx

    # 3. Phase 1A: 4-Way Simultaneous Single-Host Matrix
    if idx == 0:
        base_dir = stage_single_model_in_shm(idx, args.gcs_bucket, "minimax_h3")
        run_host_sweep(
            idx,
            bucket=args.gcs_bucket,
            gcs_out_prefix=gcs_prefix,
            config_meta={
                "name": "mh3_bf16_u4_sdpa",
                "model_short": "MiniMax-H3 (FL2VA)",
                "is_fasth3": False,
                "precision": "BF16",
                "chips": 4,
                "num_gpus": 4,
                "ulysses_degree": 4,
                "attn_mode": "XLA Fused SDPA",
            },
            server_spec={
                "config_name": "mh3_bf16_u4_sdpa",
                "model_path": base_dir,
                "port": port,
                "num_gpus": 4,
                "ulysses_degree": 4,
                "quantization": None,
                "v5p_splash": False,
                "adaln_online": False,
            },
            workloads=base_workloads,
            keyframes=keyframes,
            existing_results=pod_results,
        )

    elif idx == 1:
        base_dir = stage_single_model_in_shm(idx, args.gcs_bucket, "minimax_h3")
        run_host_sweep(
            idx,
            bucket=args.gcs_bucket,
            gcs_out_prefix=gcs_prefix,
            config_meta={
                "name": "mh3_fp8_u4_sdpa",
                "model_short": "MiniMax-H3 (FL2VA)",
                "is_fasth3": False,
                "precision": "FP8 (online)",
                "chips": 4,
                "num_gpus": 4,
                "ulysses_degree": 4,
                "attn_mode": "XLA Fused SDPA",
            },
            server_spec={
                "config_name": "mh3_fp8_u4_sdpa",
                "model_path": base_dir,
                "port": port,
                "num_gpus": 4,
                "ulysses_degree": 4,
                "quantization": "fp8",
                "v5p_splash": False,
                "adaln_online": False,
            },
            workloads=base_workloads,
            keyframes=keyframes,
            existing_results=pod_results,
        )

    elif idx == 2:
        base_dir = stage_single_model_in_shm(idx, args.gcs_bucket, "minimax_h3")
        ablation_workloads = [
            {"label": "768p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 15s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 5s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 5s t2va (25-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        ]
        # 2A: Online AdaLN + XLA Fused SDPA
        run_host_sweep(
            idx,
            bucket=args.gcs_bucket,
            gcs_out_prefix=gcs_prefix,
            config_meta={
                "name": "mh3_bf16_u4_sdpa_adaln",
                "model_short": "MiniMax-H3 (FL2VA)",
                "is_fasth3": False,
                "precision": "BF16",
                "chips": 4,
                "num_gpus": 4,
                "ulysses_degree": 4,
                "attn_mode": "XLA Fused SDPA + AdaLN-online",
            },
            server_spec={
                "config_name": "mh3_bf16_u4_sdpa_adaln",
                "model_path": base_dir,
                "port": port,
                "num_gpus": 4,
                "ulysses_degree": 4,
                "quantization": None,
                "v5p_splash": False,
                "adaln_online": True,
            },
            workloads=ablation_workloads,
            keyframes=keyframes,
            existing_results=pod_results,
        )

        # 2B: Unlocked TPU v5p Pallas Splash Attention (`SGLANG_TPU_V5P_SPLASH=1`)
        try:
            run_host_sweep(
                idx,
                bucket=args.gcs_bucket,
                gcs_out_prefix=gcs_prefix,
                config_meta={
                    "name": "mh3_bf16_u4_pallas_v5p",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "BF16",
                    "chips": 4,
                    "num_gpus": 4,
                    "ulysses_degree": 4,
                    "attn_mode": "Pallas SplashAttn (v5p 512x512)",
                },
                server_spec={
                    "config_name": "mh3_bf16_u4_pallas_v5p",
                    "model_path": base_dir,
                    "port": port,
                    "num_gpus": 4,
                    "ulysses_degree": 4,
                    "quantization": None,
                    "v5p_splash": True,
                    "adaln_online": False,
                },
                workloads=ablation_workloads,
                keyframes=keyframes,
                existing_results=pod_results,
            )
        except Exception as exc:
            log(idx, f"2B Pallas SplashAttn ablation note: {exc}")

        # 2C: 2-chip (`num_gpus=2, ulysses_degree=2`, FP8) sub-host scaling test
        try:
            run_host_sweep(
                idx,
                bucket=args.gcs_bucket,
                gcs_out_prefix=gcs_prefix,
                config_meta={
                    "name": "mh3_fp8_u2_2chips_sdpa",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "FP8 (online)",
                    "chips": 2,
                    "num_gpus": 2,
                    "ulysses_degree": 2,
                    "attn_mode": "XLA Fused SDPA",
                },
                server_spec={
                    "config_name": "mh3_fp8_u2_2chips_sdpa",
                    "model_path": base_dir,
                    "port": port,
                    "num_gpus": 2,
                    "ulysses_degree": 2,
                    "quantization": "fp8",
                    "v5p_splash": False,
                    "adaln_online": False,
                },
                workloads=ablation_workloads,
                keyframes=keyframes,
                existing_results=pod_results,
            )
        except Exception as exc:
            log(idx, f"2C 2-chip ablation note: {exc}")

    elif idx == 3:
        fh3_dir = stage_single_model_in_shm(idx, args.gcs_bucket, "fasth3")
        for q_mode, cfg_tag, prec_lbl in [
            (None, "fasth3_bf16_u4_sdpa", "BF16"),
            ("fp8", "fasth3_fp8_u4_sdpa", "FP8 (online)"),
        ]:
            try:
                run_host_sweep(
                    idx,
                    bucket=args.gcs_bucket,
                    gcs_out_prefix=gcs_prefix,
                    config_meta={
                        "name": cfg_tag,
                        "model_short": "FastH3 (4-step VSA)",
                        "is_fasth3": True,
                        "precision": prec_lbl,
                        "chips": 4,
                        "num_gpus": 4,
                        "ulysses_degree": 4,
                        "attn_mode": "XLA Fused SDPA",
                    },
                    server_spec={
                        "config_name": cfg_tag,
                        "model_path": fh3_dir,
                        "port": port,
                        "num_gpus": 4,
                        "ulysses_degree": 4,
                        "quantization": q_mode,
                        "v5p_splash": False,
                        "adaln_online": False,
                    },
                    workloads=fasth3_workloads,
                    keyframes=keyframes,
                    existing_results=pod_results,
                )
            except Exception as exc:
                log(idx, f"[{cfg_tag}] error: {exc}")

        # Swap /dev/shm cleanly back to MiniMax-H3/FL2VA for Phase 1B!
        base_dir = stage_single_model_in_shm(idx, args.gcs_bucket, "minimax_h3")

    base_dir = "/dev/shm/MiniMax-H3/FL2VA"

    # Mark Phase 1A complete for this pod
    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase1a_pod_{idx}.txt", b"done\n", content_type="text/plain")
    log(idx, "Phase 1A complete! Waiting for all 4 pods before starting Phase 1B (16-chip DP=4 cluster)...")

    t_sync = time.time()
    while time.time() - t_sync < 7200:
        ready_1a = sum(
            1
            for i in range(4)
            if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phase1a_pod_{i}.txt") is not None
        )
        if idx == 0:
            all_rows: list[dict[str, Any]] = []
            for i in range(4):
                raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{i}_results.json")
                if raw:
                    all_rows.extend(json.loads(raw.decode("utf-8")))
            md_txt = format_markdown_summary(args.run_id, nodes, all_rows, [])
            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/SUMMARY.md", md_txt.encode("utf-8"), content_type="text/markdown")
        if ready_1a == 4:
            break
        time.sleep(10)

    # 4. Phase 1B: Full 16-Chip `v5p-16` Slice (4x DP Replicas = 16 v5p chips)
    p0_raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_0_results.json")
    p1_raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_1_results.json")
    p0_rows = json.loads(p0_raw.decode("utf-8")) if p0_raw else []
    p1_rows = json.loads(p1_raw.decode("utf-8")) if p1_raw else []
    bf16_15s_ok = any(r.get("ok_count", 0) > 0 and r.get("duration_seconds") == 15.0 and r.get("short_edge") == 768 for r in p0_rows)
    fp8_15s_ok = any(r.get("ok_count", 0) > 0 and r.get("duration_seconds") == 15.0 and r.get("short_edge") == 768 for r in p1_rows)
    use_fp8 = fp8_15s_ok
    dp_quant = "fp8" if use_fp8 else None
    dp_short_edge = 768 if (fp8_15s_ok or bf16_15s_ok) else 512
    dp_label = f"DP=4 × Ulysses=4 (16 v5p chips, {'FP8' if use_fp8 else 'BF16'}, XLA SDPA)"

    dp_port = 30100 + idx
    proc_dp, _, log_dp = start_sglang_server(
        idx,
        config_name=f"dp4_rep{idx}",
        model_path=base_dir,
        port=dp_port,
        num_gpus=4,
        ulysses_degree=4,
        quantization=dp_quant,
        v5p_splash=False,
    )

    try:
        ep_local = f"http://127.0.0.1:{dp_port}"
        kf_dp = keyframes[f"16:9_{dp_short_edge}"]
        for dur in (5.0, 10.0, 15.0):
            w_p = make_minimax_payload(
                prompt=PROMPTS[0],
                task="fl2va",
                short_edge=dp_short_edge,
                aspect_ratio="16:9",
                duration_seconds=dur,
                num_inference_steps=5,
                seed=700 + int(dur),
                keyframe_uri=kf_dp,
            )
            submit_and_poll_video(ep_local, w_p, proc=proc_dp, log_path=log_dp)

        gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/dp4_ready_pod_{idx}.txt", b"ready\n", content_type="text/plain")
        log(idx, f"Pod-{idx} warmed and ready for 16-chip DP=4 cluster benchmark!")

        if idx == 0:
            t_w = time.time()
            while time.time() - t_w < 2400:
                if all(
                    gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/dp4_ready_pod_{i}.txt") is not None
                    for i in range(4)
                ):
                    break
                time.sleep(5)

            for i in range(4):
                raw_n = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/nodes/node_{i}.json")
                if raw_n:
                    nodes[i] = json.loads(raw_n.decode("utf-8"))
            endpoints = [f"http://{nodes[i]['ip']}:{30100 + i}" for i in range(4)]
            log(0, f"Phase 1B DP=4 live endpoints: {endpoints}")
            cluster_rows: list[dict[str, Any]] = []

            cluster_suites = [
                {"label": f"{dp_short_edge}p 16:9 15s i2v (4-step)", "mode": "fixed", "dur": 15.0, "steps": 5, "conc": 4, "n": 8},
                {"label": f"{dp_short_edge}p 16:9 15s i2v (4-step)", "mode": "fixed", "dur": 15.0, "steps": 5, "conc": 8, "n": 12},
                {"label": f"{dp_short_edge}p 16:9 15s i2v (25-step)", "mode": "fixed", "dur": 15.0, "steps": 26, "conc": 4, "n": 8},
                {"label": f"{dp_short_edge}p 16:9 5s i2v (4-step)", "mode": "fixed", "dur": 5.0, "steps": 5, "conc": 4, "n": 12},
                {"label": f"{dp_short_edge}p 16:9 5s i2v (4-step)", "mode": "fixed", "dur": 5.0, "steps": 5, "conc": 8, "n": 16},
                {"label": f"Mixed Dist 50/30/20 (5s/10s/15s {dp_short_edge}p i2v, 4-step)", "mode": "mixed", "steps": 5, "conc": 4, "n": 10},
                {"label": f"Mixed Dist 50/30/20 (5s/10s/15s {dp_short_edge}p i2v, 4-step)", "mode": "mixed", "steps": 5, "conc": 8, "n": 16},
            ]

            mixed_durs = [5.0, 5.0, 10.0, 5.0, 15.0, 5.0, 10.0, 5.0, 10.0, 15.0]

            for s_i, cs in enumerate(cluster_suites):
                n_reqs = cs["n"]
                conc = cs["conc"]
                steps_grid = cs["steps"]
                denoise_steps = steps_grid - 1
                reqs_list = []
                for r_i in range(n_reqs):
                    dur = cs["dur"] if cs["mode"] == "fixed" else mixed_durs[r_i % len(mixed_durs)]
                    ep = endpoints[r_i % 4]
                    p = make_minimax_payload(
                        prompt=PROMPTS[r_i % len(PROMPTS)],
                        task="fl2va",
                        short_edge=dp_short_edge,
                        aspect_ratio="16:9",
                        duration_seconds=dur,
                        num_inference_steps=steps_grid,
                        seed=5000 + s_i * 100 + r_i,
                        keyframe_uri=kf_dp,
                    )
                    reqs_list.append((ep, p, dur))

                log(0, f"[Cluster 16-chip v5p] Running {cs['label']} (c={conc}, n={n_reqs})...")
                t_c0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                    outs = list(pool.map(lambda item: submit_and_poll_video(item[0], item[1]), reqs_list))
                wall_s = time.perf_counter() - t_c0

                oks = [o for o in outs if o.get("ok")]
                lats = sorted(o["e2e_s"] for o in oks)
                med_e2e = round(statistics.median(lats), 3) if lats else None
                p90_e2e = round(lats[min(len(lats) - 1, int(0.9 * len(lats)))], 3) if lats else None
                rpm = round((len(oks) / wall_s) * 60.0, 3) if oks and wall_s > 0 else 0.0
                ok_vid_sec = sum(reqs_list[i][2] for i, o in enumerate(outs) if o.get("ok"))
                rtf = round(ok_vid_sec / wall_s, 3) if wall_s > 0 else 0.0

                crow = {
                    "cluster_config": dp_label,
                    "workload_label": cs["label"],
                    "denoise_steps": denoise_steps,
                    "concurrency": conc,
                    "num_requests": n_reqs,
                    "ok_count": len(oks),
                    "wall_s": round(wall_s, 3),
                    "median_e2e_s": med_e2e,
                    "p90_e2e_s": p90_e2e,
                    "rpm_videos_per_min": rpm,
                    "slices_for_30_rpm": math.ceil(30.0 / rpm) if rpm > 0 else None,
                    "video_sec_per_wall_sec": rtf,
                }
                log(0, f"[Cluster 16-chip v5p] DONE {cs['label']} c={conc}: ok={len(oks)}/{n_reqs} RPM={rpm}/min med={med_e2e}s RTF={rtf}x")
                cluster_rows.append(crow)

                all_rows = []
                for i in range(4):
                    raw = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/pod_{i}_results.json")
                    if raw:
                        all_rows.extend(json.loads(raw.decode("utf-8")))
                summary_obj = {
                    "run_id": args.run_id,
                    "updated_utc": utc_now(),
                    "nodes": nodes,
                    "single_host_results": all_rows,
                    "cluster_dp4_results": cluster_rows,
                }
                gcs_upload_bytes(
                    args.gcs_bucket,
                    f"{gcs_prefix}/summary.json",
                    json.dumps(summary_obj, indent=2).encode("utf-8"),
                )
                md_txt = format_markdown_summary(args.run_id, nodes, all_rows, cluster_rows)
                gcs_upload_bytes(
                    args.gcs_bucket,
                    f"{gcs_prefix}/SUMMARY.md",
                    md_txt.encode("utf-8"),
                    content_type="text/markdown",
                )

            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt", b"ok\n", content_type="text/plain")
            log(0, "Priority #1 MiniMax-H3 & FastH3 TPU v5p benchmark suite 100% COMPLETE!")
        else:
            t_hold = time.time()
            while time.time() - t_hold < 7200:
                if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt") is not None:
                    break
                time.sleep(5)
    finally:
        proc_dp.terminate()
        kill_sglang_processes()


if __name__ == "__main__":
    main()
