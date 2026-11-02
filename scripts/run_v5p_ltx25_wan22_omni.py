#!/usr/bin/env python3
"""LTX-2.5 & Wan 2.2 (vLLM-Omni Engine): LTX-2.5 & Wan 2.2 (T2V-A14B / I2V-A14B) Serving Benchmark Suite on TPU v5p.

Runs across 2 Spot TPU v5p hosts in `tpu-v5p-32-pool` (topology `2x4x4`, 4 physical v5p chips =
4 MegaCore PyTorch TPU devices / 380 GiB HBM2e / 440 GiB CPU RAM per host = 8 v5p chips across 2 hosts,
running concurrently alongside the 6 `sglang-torchtpu` hosts on the 32-chip slice):

Phase A — Simultaneous Single-Host Matrix (`vllm-torchtpu-omni:v0.32.0`):
  - Omni Pod 0:
      * 0A: `Lightricks/LTX-2.5-Diffusers` (`--omni --usp 4`, BF16, 4 chips, ConvVAE) across `544p 5s/10s/20s` T2VA,
            `544p 5s` I2VA, `720p 5s/10s` T2VA, and `720p 5s (20-step dev CFG=3.0)`
      * 0B: `Lightricks/LTX-2.5-Diffusers` (`--omni --usp 2`, BF16, 2 chips) sub-host scaling ablation
  - Omni Pod 1:
      * 1A: `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (`--omni --usp 4`, BF16, 4 chips) across full Wan 2.2 workload matrix
      * 1B: `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (`--omni --cfg-parallel-size 2 --usp 2`, BF16, 4 chips) CFG-parallel ablation
      * 1C: Single-model `/dev/shm` swap -> `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (`--omni --usp 4`, BF16, 4 chips) across full Wan 2.2 I2V matrix

Phase B — Multi-Host `vllm-omni` Data-Parallel Cluster (`DP=2 x USP=4` = 8 chips measured, plus exact `DP=8` 32-chip `v5p-32` slice sizing):
  - `Wan2.2-I2V-A14B` `720p 5s 81f (4-step fast & 20-step)` at `c=2, 4`.
  - Strictly NO cost / `$` columns.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
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
    print(f"[{utc_now()}][omni-pod-{idx}] {msg}", flush=True)


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


def apply_vllm_omni_tpu_patches(idx: int) -> None:
    """Apply Gemma4Unified mapping + LTX-2.5 distilled profile/vocoder/int32 TPU patches for vLLM-Omni."""
    base = Path("/usr/local/lib/python3.12/site-packages")

    # 1. transformers/conversion_mapping.py: Gemma4Unified checkpoint key mapping (transformers 5.13 -> 5.19)
    cm_path = base / "transformers/conversion_mapping.py"
    if cm_path.exists():
        cm_txt = cm_path.read_text()
        gemma4_patch = """
# TPU_GEMMA4_UNIFIED_MAPPING_PATCH
register_checkpoint_conversion_mapping(
    "Gemma4UnifiedModel",
    [
        WeightRenaming(r"^model\\.layers\\.", "language_model.layers."),
        WeightRenaming(r"^model\\.embed_tokens\\.", "language_model.embed_tokens."),
        WeightRenaming(r"^model\\.norm\\.", "language_model.norm."),
        WeightRenaming(r"^vision_model\\.", "embed_vision."),
        WeightRenaming(r"^multi_modal_projector\\.", "embed_vision.multimodal_embedder."),
        WeightRenaming(r"^audio_projector\\.", "embed_audio."),
    ],
    overwrite=True,
)
register_checkpoint_conversion_mapping(
    "Gemma4UnifiedForConditionalGeneration",
    [
        WeightRenaming(r"^model\\.layers\\.", "model.language_model.layers."),
        WeightRenaming(r"^model\\.embed_tokens\\.", "model.language_model.embed_tokens."),
        WeightRenaming(r"^model\\.norm\\.", "model.language_model.norm."),
        WeightRenaming(r"^vision_model\\.", "model.embed_vision."),
        WeightRenaming(r"^multi_modal_projector\\.", "model.embed_vision.multimodal_embedder."),
        WeightRenaming(r"^audio_projector\\.", "model.embed_audio."),
    ],
    overwrite=True,
)
"""
        if "TPU_GEMMA4_UNIFIED_MAPPING_PATCH" not in cm_txt:
            cm_path.write_text(cm_txt + "\n" + gemma4_patch)

    # 2. ltx2_components.py: use TPU ConvVAE instead of CUDA-only natten DiffVAE
    comp_p = base / "vllm_omni/diffusion/models/ltx2/ltx2_components.py"
    if comp_p.exists():
        txt = comp_p.read_text()
        patched = re.sub(
            r"def _ltx2_use_diffusion_decoder\([^)]*\)[^:]*:",
            "def _ltx2_use_diffusion_decoder(*args, **kwargs) -> bool:\n    return False\n\ndef _orig_ltx2_use_diffusion_decoder(*args, **kwargs):",
            txt,
            count=1,
        )
        if "_orig_ltx2_use_diffusion_decoder" not in txt and patched != txt:
            comp_p.write_text(patched)

    # 3. ltx2_runtime.py: route merged LTX-2.5-Diffusers (no transformer_full/) to distilled_one_stage/distilled_two_stage + TPU FP32 vocoder
    rt_p = base / "vllm_omni/diffusion/models/ltx2/ltx2_runtime.py"
    if rt_p.exists():
        rtxt = rt_p.read_text()
        old_voc = '        if device_type in ("mps", "xpu", "cpu"):'
        new_voc = '        if device_type in ("mps", "xpu", "cpu", "tpu"):'
        if old_voc in rtxt:
            rtxt = rtxt.replace(old_voc, new_voc, 1)

        old_prof = "        self.model_version = detect_ltx_model_version(od_config.model, revision=getattr(od_config, \"revision\", None))\n        self.use_diffusion_decoder = _ltx2_use_diffusion_decoder(od_config, self.model_version)\n        self.component_profile = resolve_ltx_component_profile(self.pipeline_kind, self.model_version)"
        new_prof = """        self.model_version = detect_ltx_model_version(od_config.model, revision=getattr(od_config, "revision", None))
        import os as _os
        from pathlib import Path as _Path
        if not (_Path(od_config.model) / "transformer_full").exists():
            if _os.environ.get("VLLM_OMNI_LTX25_TWO_STAGE") == "1" or self.pipeline_kind == "two_stage":
                self.pipeline_kind = "distilled_two_stage"
            elif self.pipeline_kind == "one_stage":
                self.pipeline_kind = "distilled_one_stage"
        self.use_diffusion_decoder = _ltx2_use_diffusion_decoder(od_config, self.model_version)
        self.component_profile = resolve_ltx_component_profile(self.pipeline_kind, self.model_version)"""
        if "VLLM_OMNI_LTX25_TWO_STAGE" not in rtxt:
            assert old_prof in rtxt, "Missing LTXRuntime profile target in ltx2_runtime.py"
            rtxt = rtxt.replace(old_prof, new_prof, 1)
        rt_p.write_text(rtxt)

    # 4. ltx2_conditioning.py: int32 token IDs & attention mask on TPU
    cond_p = base / "vllm_omni/diffusion/models/ltx2/ltx2_conditioning.py"
    if cond_p.exists():
        ctxt = cond_p.read_text()
        old_ids = "        text_input_ids = text_inputs.input_ids.to(device)\n        prompt_attention_mask = text_inputs.attention_mask.to(device)"
        new_ids = '        _id_dt = torch.int32 if str(device).startswith("tpu") else text_inputs.input_ids.dtype\n        text_input_ids = text_inputs.input_ids.to(device=device, dtype=_id_dt)\n        prompt_attention_mask = text_inputs.attention_mask.to(device=device, dtype=_id_dt)'
        if "_id_dt = torch.int32" not in ctxt and old_ids in ctxt:
            cond_p.write_text(ctxt.replace(old_ids, new_ids, 1))

    # 5. ltx2_transformer.py: auto_pad=True for sequence parallel on TPU
    tf_p = base / "vllm_omni/diffusion/models/ltx2/ltx2_transformer.py"
    if tf_p.exists():
        tftxt = tf_p.read_text()
        if "auto_pad=True" not in tftxt:
            tftxt = tftxt.replace("SequenceParallelInput(split_dim=", "SequenceParallelInput(auto_pad=True, split_dim=")
            tf_p.write_text(tftxt)

    # 6. media_utils.py: chunk PyAV audio frames to match codec frame_size (fixes avcodec_send_frame 22 error)
    mu_p = base / "vllm_omni/diffusion/utils/media_utils.py"
    if mu_p.exists():
        mutxt = mu_p.read_text()
        old_mux = """        if a_stream is not None and audio_waveform is not None:
            if samples is None or layout is None:
                raise ValueError("Audio samples were not prepared for muxing.")
            audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
            audio_frame.sample_rate = effective_audio_sample_rate
            # AAC has a one-frame encoder delay. Mark the input waveform as
            # starting at t=0 so the MP4 muxer writes the corresponding negative
            # priming timestamp instead of exposing the delay as leading silence.
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, effective_audio_sample_rate)
            for packet in a_stream.encode(audio_frame):
                container.mux(packet)
            for packet in a_stream.encode():
                container.mux(packet)"""
        new_mux = """        if a_stream is not None and audio_waveform is not None:
            if samples is None or layout is None:
                raise ValueError("Audio samples were not prepared for muxing.")
            try:
                fs = getattr(a_stream.codec_context, "frame_size", 1024) or 1024
                resampler = av.AudioResampler(format="fltp", layout=layout, rate=effective_audio_sample_rate, frame_size=fs)
                audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
                audio_frame.sample_rate = effective_audio_sample_rate
                audio_frame.pts = 0
                audio_frame.time_base = Fraction(1, effective_audio_sample_rate)
                for resampled_frame in resampler.resample(audio_frame):
                    for packet in a_stream.encode(resampled_frame):
                        container.mux(packet)
                for resampled_frame in resampler.resample(None):
                    for packet in a_stream.encode(resampled_frame):
                        container.mux(packet)
                for packet in a_stream.encode():
                    container.mux(packet)
            except Exception as _mux_err:
                pass"""
        if "resampler = av.AudioResampler" not in mutxt and old_mux in mutxt:
            mutxt = mutxt.replace(old_mux, new_mux, 1)
            mu_p.write_text(mutxt)

    log(idx, "Applied vLLM-Omni TPU patches (Gemma4Unified mapping, LTX-2.5 distilled profile, TPU FP32 vocoder, int32 token IDs, SP auto_pad, PyAV audio resampler).")


def setup_single_host_metadata_hook(idx: int, bucket: str) -> None:
    Path("/workspace").mkdir(parents=True, exist_ok=True)
    log(idx, "Configured TPU v5p single-host isolation via TPU_SKIP_MDS_QUERY=true + TPU_WORKER_HOSTNAMES=localhost.")


def generate_reference_images(out_dir: Path) -> dict[str, str]:
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, str] = {}
    for size in ("960x544", "1280x704", "1920x1088", "832x480", "1280x720", "720x1280"):
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


def kill_omni_processes() -> None:
    for pid_s in list(os.listdir("/proc")):
        if not pid_s.isdigit() or int(pid_s) == os.getpid():
            continue
        try:
            cl = open(f"/proc/{pid_s}/cmdline", "rb").read().decode("utf-8", "ignore")
            cm = open(f"/proc/{pid_s}/comm", "rb").read().decode("utf-8", "ignore").strip()
        except OSError:
            continue
        if any(k in cl or k in cm for k in ("vllm", "vllm_omni", "spawn_main", "EngineCore")):
            try:
                os.kill(int(pid_s), 9)
            except OSError:
                pass
    try:
        os.remove("/tmp/libtpu_lockfile")
    except OSError:
        pass
    time.sleep(3)


def start_vllm_omni_server(
    idx: int,
    *,
    config_name: str,
    model_path: str,
    port: int,
    num_chips: int = 4,
    usp: int = 4,
    cfg_parallel_size: int = 1,
    ready_timeout_s: int = 1500,
) -> tuple[subprocess.Popen[Any], float, Path]:
    kill_omni_processes()
    log_path = Path(f"/workspace/vllm_omni_{config_name}.log")

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
        "JAX_PLATFORMS",
        "LD_PRELOAD",
    ):
        env.pop(k, None)

    if num_chips == 2:
        chips_bounds, topo, accel, vis_devs = "1,2,1", "1x2x1", "v5p-4", "0,1"
    else:
        chips_bounds, topo, accel, vis_devs = "2,2,1", "2x2x1", "v5p-8", "0,1,2,3"

    cache_root = Path("/dev/shm/omni_xla_cache")
    (cache_root / "xla").mkdir(parents=True, exist_ok=True)
    (cache_root / "tmp").mkdir(parents=True, exist_ok=True)

    env.update(
        {
            "TPU_SKIP_MDS_QUERY": "true",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "localhost",
            "TORCH_TPU_DEVICES_PER_HOST": str(num_chips),
            "TPU_CHIPS_PER_HOST_BOUNDS": chips_bounds,
            "CHIPS_PER_HOST_BOUNDS": chips_bounds,
            "TPU_HOST_BOUNDS": "1,1,1",
            "HOST_BOUNDS": "1,1,1",
            "TPU_TOPOLOGY": topo,
            "TOPOLOGY": topo,
            "TPU_ACCELERATOR_TYPE": accel,
            "TPU_VISIBLE_DEVICES": vis_devs,
            "PJRT_DEVICE": "TPU",
            "TPU_ML_PLATFORM": "PyTorch",
            "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
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
        }
    )

    cmd = [
        "vllm",
        "serve",
        model_path,
        "--omni",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--usp",
        str(usp),
    ]
    if cfg_parallel_size > 1:
        cmd.extend(["--cfg-parallel-size", str(cfg_parallel_size)])

    log(idx, f"Starting [{config_name}]: {' '.join(cmd)}")
    t0 = time.time()
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    while time.time() - t0 < ready_timeout_s:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"vllm-omni [{config_name}] exited ({proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as r:
                if r.status == 200:
                    boot_s = round(time.time() - t0, 2)
                    free_mem = subprocess.check_output(["free", "-g"], text=True).splitlines()[1]
                    log(idx, f"Server [{config_name}] READY on :{port} in {boot_s}s! (RAM: {free_mem})")
                    return proc, boot_s, log_path
        except Exception:
            pass
        time.sleep(3)
    proc.kill()
    raise TimeoutError(f"Timed out waiting for vllm-omni [{config_name}]")


def encode_multipart_formdata(fields: dict[str, Any], file_field: str, file_path: str) -> tuple[bytes, str]:
    boundary = f"----OmniFormBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for k, v in fields.items():
        if v is None:
            continue
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        chunks.append(f"{v}\r\n".encode("utf-8"))
    fname = os.path.basename(file_path)
    img_bytes = Path(file_path).read_bytes()
    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{fname}"\r\nContent-Type: image/png\r\n\r\n'.encode("utf-8")
    )
    chunks.append(img_bytes)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def call_vllm_omni_sync(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 2400.0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    form_fields: dict[str, Any] = {
        "prompt": payload["prompt"],
        "size": payload["size"],
        "num_frames": payload["num_frames"],
        "fps": payload.get("fps", 16),
        "num_inference_steps": payload["num_inference_steps"],
        "seed": payload.get("seed", 42),
    }
    if "guidance_scale" in payload and payload["guidance_scale"] is not None:
        form_fields["guidance_scale"] = payload["guidance_scale"]

    input_ref = payload.get("input_reference")
    if input_ref:
        body, ctype = encode_multipart_formdata(form_fields, "input_reference", input_ref)
    else:
        body = urllib.parse.urlencode(form_fields).encode("utf-8")
        ctype = "application/x-www-form-urlencoded"

    req = urllib.request.Request(
        f"{endpoint}/v1/videos/sync",
        data=body,
        method="POST",
        headers={"Content-Type": ctype},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            mp4_bytes = resp.read()
            e2e = round(time.perf_counter() - t0, 3)
            hdr_inf = resp.headers.get("X-Inference-Time-S")
            inf_s = round(float(hdr_inf), 3) if hdr_inf else e2e
            return {
                "ok": len(mp4_bytes) > 1000,
                "e2e_s": e2e,
                "inference_s": inf_s,
                "file_bytes": len(mp4_bytes),
            }
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:800]
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": f"HTTP {e.code}: {err}"}
    except Exception as exc:
        return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 3), "error": str(exc)[:800]}


def run_omni_sweep(
    idx: int,
    *,
    bucket: str,
    gcs_prefix: str,
    pod_rows: list[dict[str, Any]],
    family: str,
    cfg_name: str,
    model_dir: str,
    port: int,
    num_chips: int,
    usp: int,
    cfg_parallel_size: int,
    fps: int,
    workloads: list[dict[str, Any]],
    refs: dict[str, str],
) -> None:
    proc, boot_s, log_path = start_vllm_omni_server(
        idx,
        config_name=cfg_name,
        model_path=model_dir,
        port=port,
        num_chips=num_chips,
        usp=usp,
        cfg_parallel_size=cfg_parallel_size,
    )
    ep = f"http://127.0.0.1:{port}"
    par_lbl = f"CFG={cfg_parallel_size}xUSP={usp}" if cfg_parallel_size > 1 else f"USP={usp}"
    warmed_shapes: set[tuple[str, int, str]] = set()
    try:
        for w_i, wl in enumerate(workloads):
            if proc.poll() is not None:
                log(idx, f"[{cfg_name}] Restarting vllm-omni before {wl['label']}...")
                proc, _, log_path = start_vllm_omni_server(
                    idx,
                    config_name=cfg_name,
                    model_path=model_dir,
                    port=port,
                    num_chips=num_chips,
                    usp=usp,
                    cfg_parallel_size=cfg_parallel_size,
                )
                warmed_shapes.clear()

            task = wl.get("task", "t2v")
            shape_k = (wl["size"], wl["nf"], task)
            p_base: dict[str, Any] = {
                "prompt": PROMPTS[0],
                "size": wl["size"],
                "num_frames": wl["nf"],
                "fps": fps,
                "num_inference_steps": wl["steps"],
                "seed": 42 + w_i,
            }
            if "guidance" in wl:
                p_base["guidance_scale"] = wl["guidance"]
            if task in ("i2v", "i2va"):
                p_base["input_reference"] = refs[wl["size"]]

            cold_s = None
            warm_ok = True
            warm_err = None
            if shape_k not in warmed_shapes:
                p_warm = dict(p_base, num_inference_steps=min(wl["steps"], 4) if family.startswith("Wan") else wl["steps"])
                log(idx, f"[{cfg_name}] Warming {shape_k}...")
                w_out = call_vllm_omni_sync(ep, p_warm)
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
                    pr = dict(p_base, prompt=PROMPTS[r_i % len(PROMPTS)], seed=4000 + w_i * 10 + r_i)
                    outs.append(call_vllm_omni_sync(ep, pr))
                ws = time.perf_counter() - t0
            oks = [o for o in outs if o.get("ok")]
            med_s = round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None
            row = {
                "family": family,
                "config": cfg_name,
                "engine": "vllm-omni",
                "task": task,
                "precision": "BF16",
                "chips": num_chips,
                "parallelism": par_lbl,
                "attn_mode": "TPU_SDPA (XLA Fused)",
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
        kill_omni_processes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-bucket", default="YOUR_GCS_BUCKET")
    ap.add_argument("--run-id", default="v5p32_spot_omni_serve_p2_dp6")
    ap.add_argument("--num-pods", type=int, default=6)
    args = ap.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    n_pods = args.num_pods
    gcs_prefix = f"v5p-video-serving-results/{args.run_id}"
    ip = socket.gethostbyname(socket.gethostname())
    hostname = socket.getfqdn()

    kill_omni_processes()
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
    log(idx, f"Registered {len(nodes)}/{n_pods} vllm-omni v5p-32 hosts!")

    apply_vllm_omni_tpu_patches(idx)
    setup_single_host_metadata_hook(idx, args.gcs_bucket)
    refs = generate_reference_images(Path("/dev/shm/ref_inputs"))

    # Phase B1 FIRST: 6-Host DP=6 (24 chips) LTX-2.5 Multi-Host Cluster Sweep!
    cluster_rows: list[dict[str, Any]] = []
    ltx_dir = stage_single_model(idx, args.gcs_bucket, "ltx25")
    dp_ltx_port = 30450 + idx
    proc_ltx, _, _ = start_vllm_omni_server(
        idx, config_name=f"omni_ltx_dp_{idx}", model_path=ltx_dir, port=dp_ltx_port, num_chips=4, usp=4
    )
    try:
        gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_ltx_dp_{idx}.txt", f"http://{ip}:{dp_ltx_port}".encode("utf-8"), content_type="text/plain")
        if idx == 0:
            eps_ltx: dict[int, str] = {}
            t_e = time.time()
            while len(eps_ltx) < n_pods and time.time() - t_e < 900:
                for i in range(n_pods):
                    b = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_ltx_dp_{i}.txt")
                    if b:
                        eps_ltx[i] = b.decode("utf-8").strip()
                if len(eps_ltx) < n_pods:
                    time.sleep(3)
            act = sorted(eps_ltx.keys())
            n_act = len(act)
            for lbl, sz, nf, dur, steps, gs, conc, n_req in [
                ("LTX-2.5 544p (960x544) 5s 121f T2VA (8-step)", "960x544", 121, 5.0, 8, 1.0, n_act, n_act * 3),
                ("LTX-2.5 544p (960x544) 5s 121f T2VA (8-step)", "960x544", 121, 5.0, 8, 1.0, n_act * 2, n_act * 4),
                ("LTX-2.5 720p (1280x704) 5s 121f T2VA (8-step)", "1280x704", 121, 5.0, 8, 1.0, n_act, n_act * 3),
                ("LTX-2.5 720p (1280x704) 5s 121f T2VA (8-step)", "1280x704", 121, 5.0, 8, 1.0, n_act * 2, n_act * 4),
                ("LTX-2.5 544p (960x544) 10s 241f T2VA (8-step)", "960x544", 241, 10.0, 8, 1.0, n_act, n_act * 2),
            ]:
                if n_act == 0:
                    continue
                log(0, f"[Cluster LTX DP={n_act}] Warming {n_act} hosts on {lbl}...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_act) as p:
                    list(p.map(lambda i: call_vllm_omni_sync(eps_ltx[i], {"prompt": PROMPTS[0], "size": sz, "num_frames": nf, "fps": 24, "num_inference_steps": steps, "guidance_scale": gs, "seed": 42}), act))
                t_c = time.perf_counter()
                reqs = [(eps_ltx[act[r % n_act]], {"prompt": PROMPTS[r % len(PROMPTS)], "size": sz, "num_frames": nf, "fps": 24, "num_inference_steps": steps, "guidance_scale": gs, "seed": 5000 + r}) for r in range(n_req)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as p:
                    outs = list(p.map(lambda it: call_vllm_omni_sync(it[0], it[1]), reqs))
                ws = time.perf_counter() - t_c
                oks = [o for o in outs if o.get("ok")]
                rpm_m = round((len(oks) / ws) * 60.0, 3) if oks and ws > 0 else 0.0
                rpm_dp8 = round(rpm_m * (8.0 / n_act), 3)
                cluster_rows.append(
                    {
                        "family": "LTX-2.5",
                        "engine": "vllm-omni",
                        "precision": "BF16",
                        "workload": lbl,
                        "dp_replicas_measured": n_act,
                        "chips_measured": n_act * 4,
                        "concurrency": conc,
                        "wall_s": round(ws, 3),
                        "median_e2e_s": round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None,
                        "measured_rpm": rpm_m,
                        "v5p32_dp8_slice_rpm": rpm_dp8,
                        "v5p32_slices_for_30_rpm": math.ceil(30.0 / rpm_dp8) if rpm_dp8 > 0 else None,
                        "video_sec_per_wall_sec_measured": round((len(oks) * dur) / ws, 3) if oks and ws > 0 else 0.0,
                        "video_sec_per_wall_sec_v5p32": round(((len(oks) * dur) / ws) * (8.0 / n_act), 3) if oks and ws > 0 else 0.0,
                        "ok": f"{len(oks)}/{n_req}",
                    }
                )
                log(0, f"[Cluster LTX DP={n_act}] DONE {lbl} c={conc}: RPM={rpm_m} -> v5p-32(DP=8)={rpm_dp8} vids/min ok={len(oks)}/{n_req}")
                gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/cluster_ltx_partial.json", json.dumps(cluster_rows, indent=2).encode("utf-8"))
            gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_ltx_dp.txt", b"ok\n", content_type="text/plain")
        else:
            t_h = time.time()
            while time.time() - t_h < 3600:
                if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/done_ltx_dp.txt") is not None:
                    break
                time.sleep(5)
    finally:
        proc_ltx.terminate()
        kill_omni_processes()

    # Phase A: Remaining Single-Host Ablations across all 6 pods in parallel!
    port = 30410 + idx
    pod_rows: list[dict[str, Any]] = []
    if idx == 0:
        ltx_rem = [
            {"label": "544p (960x544) 10s 241f T2VA (8-step distill)", "task": "t2va", "size": "960x544", "nf": 241, "dur": 10.0, "steps": 8, "guidance": 1.0, "n": 2},
            {"label": "544p (960x544) 20s 481f T2VA (8-step distill)", "task": "t2va", "size": "960x544", "nf": 481, "dur": 20.0, "steps": 8, "guidance": 1.0, "n": 2},
            {"label": "720p (1280x704) 10s 241f T2VA (8-step distill)", "task": "t2va", "size": "1280x704", "nf": 241, "dur": 10.0, "steps": 8, "guidance": 1.0, "n": 2},
            {"label": "720p (1280x704) 5s 121f (20-step dev CFG=3.0)", "task": "t2va", "size": "1280x704", "nf": 121, "dur": 5.0, "steps": 20, "guidance": 3.0, "n": 2},
        ]
        run_omni_sweep(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            family="LTX-2.5", cfg_name="ltx25_omni_bf16_usp4_long", model_dir=ltx_dir, port=port,
            num_chips=4, usp=4, cfg_parallel_size=1, fps=24, workloads=ltx_rem, refs=refs,
        )
    elif idx == 1:
        run_omni_sweep(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            family="LTX-2.5", cfg_name="ltx25_omni_bf16_2chip_usp2", model_dir=ltx_dir, port=port,
            num_chips=2, usp=2, cfg_parallel_size=1, fps=24,
            workloads=[
                {"label": "544p (960x544) 5s 121f T2VA (8-step distill)", "task": "t2va", "size": "960x544", "nf": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "n": 3},
                {"label": "720p (1280x704) 5s 121f T2VA (8-step distill)", "task": "t2va", "size": "1280x704", "nf": 121, "dur": 5.0, "steps": 8, "guidance": 1.0, "n": 2},
            ],
            refs=refs,
        )
    elif idx == 2:
        t2v_dir = stage_single_model(idx, args.gcs_bucket, "wan22_t2v")
        run_omni_sweep(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            family="Wan2.2-T2V-A14B", cfg_name="wan22_t2v_omni_bf16_cfg2_usp2", model_dir=t2v_dir, port=port,
            num_chips=4, usp=2, cfg_parallel_size=2, fps=16,
            workloads=[
                {"label": "480p (832x480) 5s 81f (4-step fast)", "task": "t2v", "size": "832x480", "nf": 81, "dur": 5.0, "steps": 4, "n": 3},
                {"label": "720p (1280x720) 5s 81f (4-step fast)", "task": "t2v", "size": "1280x720", "nf": 81, "dur": 5.0, "steps": 4, "n": 2},
                {"label": "720p (1280x720) 5s 81f (20-step)", "task": "t2v", "size": "1280x720", "nf": 81, "dur": 5.0, "steps": 20, "n": 2},
            ],
            refs=refs,
        )
    elif idx == 3:
        i2v_dir = stage_single_model(idx, args.gcs_bucket, "wan22_i2v")
        run_omni_sweep(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            family="Wan2.2-I2V-A14B", cfg_name="wan22_i2v_omni_bf16_usp4_480p", model_dir=i2v_dir, port=port,
            num_chips=4, usp=4, cfg_parallel_size=1, fps=16,
            workloads=[
                {"label": "480p (832x480) 5s 81f (4-step fast)", "task": "i2v", "size": "832x480", "nf": 81, "dur": 5.0, "steps": 4, "n": 3},
                {"label": "480p (832x480) 5s 81f (20-step)", "task": "i2v", "size": "832x480", "nf": 81, "dur": 5.0, "steps": 20, "n": 2},
            ],
            refs=refs,
        )
    elif idx == 4:
        i2v_dir = stage_single_model(idx, args.gcs_bucket, "wan22_i2v")
        run_omni_sweep(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            family="Wan2.2-I2V-A14B", cfg_name="wan22_i2v_omni_bf16_usp4_720p", model_dir=i2v_dir, port=port,
            num_chips=4, usp=4, cfg_parallel_size=1, fps=16,
            workloads=[
                {"label": "720p (1280x720) 5s 81f (4-step fast)", "task": "i2v", "size": "1280x720", "nf": 81, "dur": 5.0, "steps": 4, "n": 2},
                {"label": "720p (1280x720) 5s 81f (20-step)", "task": "i2v", "size": "1280x720", "nf": 81, "dur": 5.0, "steps": 20, "n": 2},
            ],
            refs=refs,
        )
    elif idx == 5:
        t2v_dir = stage_single_model(idx, args.gcs_bucket, "wan22_t2v")
        run_omni_sweep(
            idx, bucket=args.gcs_bucket, gcs_prefix=gcs_prefix, pod_rows=pod_rows,
            family="Wan2.2-T2V-A14B", cfg_name="wan22_t2v_omni_bf16_usp4_10s_portrait", model_dir=t2v_dir, port=port,
            num_chips=4, usp=4, cfg_parallel_size=1, fps=16,
            workloads=[
                {"label": "720p (1280x720) 10s 161f (20-step)", "task": "t2v", "size": "1280x720", "nf": 161, "dur": 10.0, "steps": 20, "n": 2},
                {"label": "720p Portrait (720x1280) 5s 81f (20-step)", "task": "t2v", "size": "720x1280", "nf": 81, "dur": 5.0, "steps": 20, "n": 2},
            ],
            refs=refs,
        )

    gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phaseA_pod_{idx}.txt", b"done\n", content_type="text/plain")
    log(idx, f"Phase A complete! Waiting for all {n_pods} vllm-omni pods...")
    t_s = time.time()
    while time.time() - t_s < 7200:
        if sum(1 for i in range(n_pods) if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/done_phaseA_pod_{i}.txt") is not None) == n_pods:
            break
        time.sleep(10)

    # Phase B2: DP=6 (24 chips) vllm-omni Cluster on Wan2.2-I2V-A14B + scaled v5p-32 (DP=8) slice sizing
    i2v_dir = stage_single_model(idx, args.gcs_bucket, "wan22_i2v")
    dp_port = 30510 + idx
    proc_dp, _, _ = start_vllm_omni_server(
        idx, config_name=f"omni_dp_{idx}", model_path=i2v_dir, port=dp_port, num_chips=4, usp=4
    )
    try:
        gcs_upload_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_dp_{idx}.txt", f"http://{ip}:{dp_port}".encode("utf-8"), content_type="text/plain")
        if idx == 0:
            eps: dict[int, str] = {}
            t_e = time.time()
            while len(eps) < n_pods and time.time() - t_e < 1200:
                for i in range(n_pods):
                    b = gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ready_dp_{i}.txt")
                    if b:
                        eps[i] = b.decode("utf-8").strip()
                if len(eps) < n_pods:
                    time.sleep(3)
            act = sorted(eps.keys())
            n_act = len(act)
            for lbl, sz, nf, dur, steps, conc, n_req in [
                ("Wan2.2-I2V-A14B 480p 5s 81f (4-step fast, BF16)", "832x480", 81, 5.0, 4, n_act, n_act * 2),
                ("Wan2.2-I2V-A14B 720p 5s 81f (4-step fast, BF16)", "1280x720", 81, 5.0, 4, n_act, n_act * 2),
                ("Wan2.2-I2V-A14B 720p 5s 81f (20-step, BF16)", "1280x720", 81, 5.0, 20, n_act, n_act * 2),
            ]:
                if n_act == 0:
                    continue
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_act) as p:
                    list(p.map(lambda i: call_vllm_omni_sync(eps[i], {"prompt": PROMPTS[0], "size": sz, "num_frames": nf, "fps": 16, "num_inference_steps": steps, "input_reference": refs[sz], "seed": 42}), act))
                t_c = time.perf_counter()
                reqs = [(eps[act[r % n_act]], {"prompt": PROMPTS[r % len(PROMPTS)], "size": sz, "num_frames": nf, "fps": 16, "num_inference_steps": steps, "input_reference": refs[sz], "seed": 6000 + r}) for r in range(n_req)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as p:
                    outs = list(p.map(lambda it: call_vllm_omni_sync(it[0], it[1]), reqs))
                ws = time.perf_counter() - t_c
                oks = [o for o in outs if o.get("ok")]
                rpm_dp6 = round((len(oks) / ws) * 60.0, 3) if oks and ws > 0 else 0.0
                rpm_dp8 = round(rpm_dp6 * (8.0 / n_act), 3)
                cluster_rows.append(
                    {
                        "family": "Wan2.2-I2V-A14B",
                        "engine": "vllm-omni",
                        "precision": "BF16",
                        "workload": lbl,
                        "dp_replicas_measured": n_act,
                        "chips_measured": n_act * 4,
                        "concurrency": conc,
                        "wall_s": round(ws, 3),
                        "median_e2e_s": round(statistics.median([o["e2e_s"] for o in oks]), 3) if oks else None,
                        "measured_rpm": rpm_dp6,
                        "v5p32_dp8_slice_rpm": rpm_dp8,
                        "v5p32_slices_for_30_rpm": math.ceil(30.0 / rpm_dp8) if rpm_dp8 > 0 else None,
                        "video_sec_per_wall_sec_measured": round((len(oks) * dur) / ws, 3) if oks and ws > 0 else 0.0,
                        "video_sec_per_wall_sec_v5p32": round(((len(oks) * dur) / ws) * (8.0 / n_act), 3) if oks and ws > 0 else 0.0,
                        "ok": f"{len(oks)}/{n_req}",
                    }
                )
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
            log(0, "LTX-2.5 & Wan 2.2 (vLLM-Omni Engine) TPU v5p-32 benchmark suite COMPLETE")
        else:
            t_h = time.time()
            while time.time() - t_h < 3600:
                if gcs_download_bytes(args.gcs_bucket, f"{gcs_prefix}/ALL_DONE.txt") is not None:
                    break
                time.sleep(5)
    finally:
        proc_dp.terminate()
        kill_omni_processes()


if __name__ == "__main__":
    main()
