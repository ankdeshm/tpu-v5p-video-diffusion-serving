#!/usr/bin/env python3
"""Priority #1: MiniMax-H3 & FastH3 (4-step VSA Distilled) Video+Audio Serving Benchmark Suite on TPU v5p.

Runs across the 4 Spot TPU v5p hosts in `tpu-v5p-16-pool` (topology `2x2x4`, 4 physical v5p chips =
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
    """Apply all 9 verified SGLang TPU runtime patches for MiniMax-H3 & FastH3 + SGLANG_TPU_V5P_SPLASH unlock."""
    sp = Path("/usr/local/lib/python3.12/site-packages")
    rt = sp / "sglang/multimodal_gen/runtime"

    # 1. minimax_h3_adaln_cache.py: TPU int32 collective probe + CPU safe_open + int32 resolve_slots
    p1 = rt / "models/dits/minimax_h3_adaln_cache.py"
    t1 = p1.read_text()
    old1 = """    probe = torch.tensor(
        [value],
        dtype=torch.int64,
        device=device if device.type == "cuda" else "cpu",
    )
    probe = get_world_group().all_reduce(probe, op=torch.distributed.ReduceOp.MIN)"""
    new1 = """    from sglang.multimodal_gen.runtime.platforms import current_platform
    _coll_dev = (
        current_platform.get_local_torch_device()
        if current_platform.device_type == "tpu"
        else (device if device.type == "cuda" else "cpu")
    )
    probe = torch.tensor(
        [value],
        dtype=torch.int32 if current_platform.device_type == "tpu" else torch.int64,
        device=_coll_dev,
    )
    probe = get_world_group().all_reduce(probe, op=torch.distributed.ReduceOp.MIN)"""
    if new1 not in t1:
        assert old1 in t1, "Patch 1a target missing in minimax_h3_adaln_cache.py"
        t1 = t1.replace(old1, new1, 1)

    old1b = """            handles = [
                stack.enter_context(safe_open(f, framework="pt", device=str(device)))
                for f in self.weight_files
            ]
            index = {name: h for h in handles for name in h.keys()}

            def read_shard(name: str, out_features: int) -> torch.Tensor:
                if tp_size == 1:
                    tensor = index[name].get_tensor(name)
                else:
                    shard = out_features // tp_size
                    start = tp_rank * shard
                    tensor = index[name].get_slice(name)[start : start + shard]
                if self.precision == "fp32":
                    tensor = tensor.float()
                return tensor"""
    new1b = """            handles = [
                stack.enter_context(safe_open(f, framework="pt", device="cpu"))
                for f in self.weight_files
            ]
            index = {name: h for h in handles for name in h.keys()}

            def read_shard(name: str, out_features: int) -> torch.Tensor:
                if tp_size == 1:
                    tensor = index[name].get_tensor(name)
                else:
                    shard = out_features // tp_size
                    start = tp_rank * shard
                    tensor = index[name].get_slice(name)[start : start + shard]
                if self.precision == "fp32":
                    tensor = tensor.float()
                return tensor.to(device)"""
    if new1b not in t1:
        assert old1b in t1, "Patch 1b target missing in minimax_h3_adaln_cache.py"
        t1 = t1.replace(old1b, new1b, 1)

    old1c = "        return torch.tensor(slots, dtype=torch.int64, device=self.block_params.device)"
    new1c = "        return torch.tensor(slots, dtype=torch.int32 if self.block_params.device.type == 'tpu' else torch.int64, device=self.block_params.device)"
    if new1c not in t1:
        assert old1c in t1, "Patch 1c target missing in minimax_h3_adaln_cache.py"
        t1 = t1.replace(old1c, new1c, 1)
    p1.write_text(t1)

    # 2. minimax_h3_vdn_attention.py: guard CUDA stream overlap behind q.is_cuda
    p2 = rt / "models/dits/minimax_h3_vdn_attention.py"
    t2 = p2.read_text()
    old2 = """    main_stream = torch.cuda.current_stream(q.device)
    side_stream = _linear_branch_stream(q.device)
    side_stream.wait_stream(main_stream)
    softmax_out = softmax()
    softmax_work, softmax_recv = a2a_back(softmax_out, role="vdn_out0")
    with torch.cuda.stream(side_stream):
        linear_work, linear_recv = a2a_back(linear_branch(), role="vdn_out1")
    main_stream.wait_stream(side_stream)
    softmax_work.wait()
    linear_work.wait()"""
    new2 = """    if q.is_cuda:
        main_stream = torch.cuda.current_stream(q.device)
        side_stream = _linear_branch_stream(q.device)
        side_stream.wait_stream(main_stream)
        softmax_out = softmax()
        softmax_work, softmax_recv = a2a_back(softmax_out, role="vdn_out0")
        with torch.cuda.stream(side_stream):
            linear_work, linear_recv = a2a_back(linear_branch(), role="vdn_out1")
        main_stream.wait_stream(side_stream)
        softmax_work.wait()
        linear_work.wait()
    else:
        softmax_out = softmax()
        softmax_work, softmax_recv = a2a_back(softmax_out, role="vdn_out0")
        linear_work, linear_recv = a2a_back(linear_branch(), role="vdn_out1")
        softmax_work.wait()
        linear_work.wait()"""
    if new2 not in t2:
        assert old2 in t2, "Patch 2 target missing in minimax_h3_vdn_attention.py"
        p2.write_text(t2.replace(old2, new2, 1))

    # 3. minimax_h3_qwen3vl.py: always compute get_rope_index on CPU and pass int32 indices to TPU
    p3 = rt / "models/encoders/minimax_h3_qwen3vl.py"
    t3 = p3.read_text()
    old3 = """        position_ids = None
        if host_image_grid_thw is not None or host_video_grid_thw is not None:
            position_ids, _ = self.model.get_rope_index(
                host_ids,
                host_image_grid_thw,
                host_video_grid_thw,
                attention_mask=torch.ones_like(host_ids),
            )
        ids = host_ids.to(self.device)
        call_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }
        if position_ids is not None:
            call_kwargs["position_ids"] = position_ids.to(self.device)"""
    new3 = """        position_ids, _ = self.model.get_rope_index(
            host_ids,
            host_image_grid_thw,
            host_video_grid_thw,
            attention_mask=torch.ones_like(host_ids),
        )
        self.model._tpu_host_ids = host_ids
        idx_dtype = torch.int32 if self.device.type == "tpu" else torch.long
        ids = host_ids.to(device=self.device, dtype=idx_dtype)
        call_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "position_ids": position_ids.to(device=self.device, dtype=idx_dtype),
        }"""
    if new3 not in t3:
        assert old3 in t3, "Patch 3 target missing in minimax_h3_qwen3vl.py"
        p3.write_text(t3.replace(old3, new3, 1))

    # 4. qwen3vl.py: static torch.cat splice for image_embeds and _deepstack_process on TPU
    p4 = rt / "models/encoders/qwen3vl.py"
    t4 = p4.read_text()
    old4a = """    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :]
        local_this.add_(visual_embeds)
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states"""
    new4a = """    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        if isinstance(visual_pos_masks, tuple):
            s, e = visual_pos_masks
            return torch.cat(
                [
                    hidden_states[:, :s, :],
                    hidden_states[:, s:e, :] + visual_embeds.unsqueeze(0),
                    hidden_states[:, e:, :],
                ],
                dim=1,
            )
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        local_this = hidden_states[visual_pos_masks, :]
        local_this.add_(visual_embeds)
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states"""
    if new4a not in t4:
        assert old4a in t4, "Patch 4a target missing in qwen3vl.py"
        t4 = t4.replace(old4a, new4a, 1)

    old4b = """        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self._get_flat_visual_features(
                pixel_values, image_grid_thw
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            if inputs_embeds_owned:
                inputs_embeds.masked_scatter_(image_mask, image_embeds)
            else:
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                inputs_embeds_owned = True

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self._get_flat_visual_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            if inputs_embeds_owned:
                inputs_embeds.masked_scatter_(video_mask, video_embeds)
            else:
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                inputs_embeds_owned = True

        visual_pos_masks = None
        deepstack_visual_embeds = None"""
    new4b = """        image_mask = None
        video_mask = None
        _tpu_visual_slice = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self._get_flat_visual_features(
                pixel_values, image_grid_thw
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            if (
                inputs_embeds.device.type == "tpu"
                and getattr(self, "_tpu_host_ids", None) is not None
                and inputs_embeds.shape[0] == 1
                and pixel_values_videos is None
            ):
                _h_ids = self._tpu_host_ids[0]
                _pos = (_h_ids == self.config.image_token_id).nonzero(as_tuple=False).squeeze(-1)
                _s = int(_pos[0].item())
                _e = int(_pos[-1].item()) + 1
                inputs_embeds = torch.cat(
                    [inputs_embeds[:, :_s, :], image_embeds.unsqueeze(0), inputs_embeds[:, _e:, :]],
                    dim=1,
                )
                inputs_embeds_owned = True
                _tpu_visual_slice = (_s, _e)
            else:
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                if inputs_embeds_owned:
                    inputs_embeds.masked_scatter_(image_mask, image_embeds)
                else:
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                    inputs_embeds_owned = True

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self._get_flat_visual_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            if inputs_embeds_owned:
                inputs_embeds.masked_scatter_(video_mask, video_embeds)
            else:
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                inputs_embeds_owned = True

        visual_pos_masks = _tpu_visual_slice
        deepstack_visual_embeds = deepstack_image_embeds if _tpu_visual_slice is not None else None"""
    if new4b not in t4:
        assert old4b in t4, "Patch 4b target missing in qwen3vl.py"
        t4 = t4.replace(old4b, new4b, 1)
    p4.write_text(t4)

    # 5. denoising.py: allow tpu platform in minimax_h3 full-loop denoise stage
    p5 = rt / "pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py"
    t5 = p5.read_text()
    old5 = """        if not (
            current_platform.is_cuda()
            or current_platform.is_hip()
            or current_platform.is_cpu()
            or current_platform.is_mps()
            or current_platform.is_npu()
            or current_platform.is_xpu()
        ):"""
    new5 = """        if not (
            current_platform.is_cuda()
            or current_platform.is_hip()
            or current_platform.is_cpu()
            or current_platform.is_mps()
            or current_platform.is_npu()
            or current_platform.is_xpu()
            or current_platform.device_type == "tpu"
        ):"""
    if new5 not in t5:
        assert old5 in t5, "Patch 5 target missing in minimax_h3/stages/denoising.py"
        p5.write_text(t5.replace(old5, new5, 1))

    # 6. srt/layers/rotary_embedding/mrope.py: replace step=3 strided slice_scatter with XLA-safe torch.where
    p6 = sp / "sglang/srt/layers/rotary_embedding/mrope.py"
    t6 = p6.read_text()
    old6 = """def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t"""
    new6 = """def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    num_pairs = x.shape[-1]
    lim1 = min(3 * mrope_section[1], num_pairs)
    lim2 = min(3 * mrope_section[2], num_pairs)
    lanes = torch.arange(num_pairs, device=x.device, dtype=torch.int32)
    m1 = ((lanes % 3) == 1) & (lanes < lim1)
    m2 = ((lanes % 3) == 2) & (lanes < lim2)
    out = torch.where(m1, x[1], x[0])
    out = torch.where(m2, x[2], out)
    return out"""
    if new6 not in t6:
        assert old6 in t6, "Patch 6 target missing in srt/layers/rotary_embedding/mrope.py"
        p6.write_text(t6.replace(old6, new6, 1))

    # 7. srt/layers/quantization/fp8_utils.py: TPU XLA torch._scaled_mm instead of CUDA Triton kernel
    p7 = sp / "sglang/srt/layers/quantization/fp8_utils.py"
    t7 = p7.read_text()
    old7 = """    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:"""
    new7 = """    if input.device.type == "tpu":
        input_2d = input.reshape(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[1]]
        if input_2d.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            qinput = input_2d
            x_scale = (
                input_scale.float()
                if input_scale is not None
                else torch.ones(1, device=input.device, dtype=torch.float32)
            )
            out_dtype = pre_quant_output_dtype or torch.bfloat16
        elif input_scale is not None:
            fmax = torch.finfo(torch.float8_e4m3fn).max
            qinput = (input_2d.float() / input_scale.float()).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
            x_scale = input_scale.float()
            out_dtype = input.dtype
        else:
            qinput, x_scale = input_to_float8(input_2d, dtype=torch.float8_e4m3fn)
            out_dtype = input.dtype
        if weight_scale.numel() == 1:
            out = torch._scaled_mm(
                qinput,
                weight,
                scale_a=x_scale.reshape(1),
                scale_b=weight_scale.float().reshape(1),
                out_dtype=out_dtype,
                bias=bias,
            )
            return out.view(*output_shape)
        out = torch._scaled_mm(
            qinput,
            weight,
            scale_a=torch.ones(1, device=input.device, dtype=torch.float32),
            scale_b=torch.ones(1, device=input.device, dtype=torch.float32),
            out_dtype=torch.float32,
        )
        out = out * x_scale.reshape(-1, 1) * weight_scale.float().reshape(1, -1)
        if bias is not None:
            out = out + bias.float()
        return out.to(out_dtype).view(*output_shape)

    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:"""
    if 'if input.device.type == "tpu":' not in t7:
        assert old7 in t7, "Patch 7 target missing in srt/layers/quantization/fp8_utils.py"
        p7.write_text(t7.replace(old7, new7, 1))

    # 8. sglang_tpu/multimodal_gen/tpu_splash_attn.py: SGLANG_TPU_V5P_SPLASH + SGLANG_TPU_SPLASH_VARLEN + SGLANG_TPU_DISABLE_SPLASH
    sp_path = sp / "sglang_tpu/multimodal_gen/tpu_splash_attn.py"
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

    old8a = "        self.use_splash = self.dropout == 0.0 and _check_splash_available()"
    new8a = (
        "        import os\n"
        "        self.use_splash = (\n"
        '            os.environ.get("SGLANG_TPU_DISABLE_SPLASH", "0") != "1"\n'
        "            and self.dropout == 0.0\n"
        "            and _check_splash_available()\n"
        "        )"
    )
    if old8a in stxt and "SGLANG_TPU_DISABLE_SPLASH" not in stxt:
        stxt = stxt.replace(old8a, new8a, 1)

    target8 = "        return self._sdpa.forward_varlen("
    repl8 = """        import os
        if self.use_splash and os.environ.get("SGLANG_TPU_SPLASH_VARLEN", "0") == "1":
            return super().forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        return self._sdpa.forward_varlen("""
    if target8 in stxt and "SGLANG_TPU_SPLASH_VARLEN" not in stxt:
        stxt = stxt.replace(target8, repl8, 1)
    sp_path.write_text(stxt)

    # 9. minimax_h3/stages/decoding.py: symmetric CPU AudioVAE decode on TPU to avoid rank-0-only PJRT barrier hang
    p9 = rt / "pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py"
    t9 = p9.read_text()
    old9 = """        # Audio VAE weights are replicated. Decode on replica rank 0 and broadcast
        # only within the request's replica, excluding independent DP replicas.
        replica_group = get_replica_group() if model_parallel_is_initialized() else None
        is_audio_owner = replica_group is None or replica_group.rank_in_group == 0
        owner_exception = None
        owner_error = None
        audio_payload = None
        if is_audio_owner:
            try:
                audio_payload = self._decode_audio(audio_latent, server_args)
            except Exception as exc:
                owner_exception = exc
                owner_error = f"{type(exc).__name__}: {exc}"
        if replica_group is not None:
            owner_error = replica_group.broadcast_object(owner_error, src=0)
        if owner_error is not None:
            if owner_exception is not None:
                raise owner_exception
            raise RuntimeError(
                f"MiniMax H3 audio decode failed on rank 0: {owner_error}"
            )
        if replica_group is not None:
            audio_payload = replica_group.broadcast_tensor_dict(audio_payload, src=0)"""
    new9 = """        replica_group = get_replica_group() if model_parallel_is_initialized() else None
        is_audio_owner = replica_group is None or replica_group.rank_in_group == 0
        owner_exception = None
        owner_error = None
        audio_payload = None
        _is_tpu_decode = visual_latent.device.type == "tpu"
        if _is_tpu_decode and self.audio_vae is not None:
            _avae_param = next(self.audio_vae.parameters(), None)
            if _avae_param is not None and _avae_param.device.type == "tpu":
                self.audio_vae = self.audio_vae.to(device="cpu", dtype=torch.float32)
        _audio_latent_cpu = (
            audio_latent.detach().to(device="cpu", dtype=torch.float32)
            if _is_tpu_decode
            else None
        )
        if is_audio_owner:
            try:
                if _is_tpu_decode:
                    assert self.audio_vae is not None
                    _avae = self.audio_vae.to(device="cpu", dtype=torch.float32)
                    if _avae.training:
                        _avae.eval()
                    _a_cfg = server_args.pipeline_config.audio_vae_config.arch_config
                    _a_lat = _reverse_normalize_latents(
                        _audio_latent_cpu,
                        mean_values=_a_cfg.latents_mean,
                        std_values=_a_cfg.latents_std,
                        name="audio_vae",
                    ).to(device="cpu", dtype=torch.float32)
                    _wf = _required_tensor(_avae.decode(_a_lat), "audio_vae.decode")
                    audio_payload = {
                        "waveform": _wf.detach().to(device="cpu", dtype=torch.float32).contiguous(),
                        "sample_rate": int(_avae.sample_rate),
                    }
                else:
                    audio_payload = self._decode_audio(audio_latent, server_args)
            except Exception as exc:
                owner_exception = exc
                owner_error = f"{type(exc).__name__}: {exc}"
        if replica_group is not None:
            owner_error = replica_group.broadcast_object(owner_error, src=0)
        if owner_error is not None:
            if owner_exception is not None:
                raise owner_exception
            raise RuntimeError(
                f"MiniMax H3 audio decode failed on rank 0: {owner_error}"
            )
        if replica_group is not None:
            if _is_tpu_decode:
                audio_payload = replica_group.broadcast_object(audio_payload, src=0)
            else:
                audio_payload = replica_group.broadcast_tensor_dict(audio_payload, src=0)"""
    if '_is_tpu_decode = visual_latent.device.type == "tpu"' not in t9:
        assert old9 in t9, "Patch 9 target missing in minimax_h3/stages/decoding.py"
        p9.write_text(t9.replace(old9, new9, 1))

    log(idx, "Applied all 9 SGLang TPU v5p runtime patches (AdaLN int32, VDN, Qwen3-VL, Denoise, MRoPE, FP8 _scaled_mm, v5p Splash, CPU AudioDecode).")


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


def ensure_fl2va_symlinks(fl2va_dir: Path) -> None:
    if not (fl2va_dir / "vae").exists() and (fl2va_dir / "video_vae").exists():
        (fl2va_dir / "vae").symlink_to("video_vae")
    v_sf = fl2va_dir / "video_vae" / "model.safetensors"
    v_src = fl2va_dir / "video_vae" / "source" / "model.safetensors"
    if not v_sf.exists() and v_src.exists():
        v_sf.symlink_to("source/model.safetensors")


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
                "models/MiniMaxAI/MiniMax-H3/FL2VA/",
                base_dir / "FL2VA",
            )
            ensure_fl2va_symlinks(base_dir / "FL2VA")
            assert (base_dir / "FL2VA" / "model_index.json").exists(), "Missing FL2VA/model_index.json!"
            marker.write_text("ok\n")
        else:
            ensure_fl2va_symlinks(base_dir / "FL2VA")
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
    cfg_parallel_size: int = 1,
    adaln_online: bool = True,
    encoder_offload: bool = True,
    ready_timeout_s: int = 1200,
) -> tuple[subprocess.Popen[Any], float, Path]:
    kill_sglang_processes()
    out_dir = Path(f"/dev/shm/sglang_out_{config_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"/workspace/sglang_{config_name}.log")

    off_val = "true" if encoder_offload else "false"
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
        f"--text-encoder-cpu-offload={off_val}",
        f"--image-encoder-cpu-offload={off_val}",
        f"--vae-cpu-offload={off_val}",
        f"--offload-during-compile={off_val}",
        "--output-path",
        str(out_dir),
    ]
    if cfg_parallel_size > 1:
        cmd.extend(["--cfg-parallel-size", str(cfg_parallel_size)])
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
    timeout_s: float = 2400.0,
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
        with urllib.request.urlopen(req, timeout=600) as resp:
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
        "- **Cluster / Nodepool:** `tpu-v5p-cluster` / `tpu-v5p-16-pool` (`2x2x4`, `4` hosts × `4` physical `v5p` chips = `16` physical chips / `16` MegaCore devices / `1,520 GiB` HBM2e)",
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
    ap.add_argument("--gcs-bucket", default="YOUR_GCS_BUCKET")
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
                "adaln_online": True,
            },
            workloads=base_workloads,
            keyframes=keyframes,
            existing_results=pod_results,
        )

    elif idx == 1:
        base_dir = stage_single_model_in_shm(idx, args.gcs_bucket, "minimax_h3")
        cfg2_workloads = [
            {"label": "768p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 15s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 5s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 5s t2va (25-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
            {"label": "512p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 512, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
        ]
        try:
            run_host_sweep(
                idx,
                bucket=args.gcs_bucket,
                gcs_out_prefix=gcs_prefix,
                config_meta={
                    "name": "mh3_bf16_cfg2_u2_sdpa",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "BF16",
                    "chips": 4,
                    "num_gpus": 4,
                    "ulysses_degree": 2,
                    "attn_mode": "CFG=2 × USP=2 + XLA SDPA + AdaLN-online",
                },
                server_spec={
                    "config_name": "mh3_bf16_cfg2_u2_sdpa",
                    "model_path": base_dir,
                    "port": port,
                    "num_gpus": 4,
                    "ulysses_degree": 2,
                    "cfg_parallel_size": 2,
                    "quantization": None,
                    "v5p_splash": False,
                    "adaln_online": True,
                },
                workloads=cfg2_workloads,
                keyframes=keyframes,
                existing_results=pod_results,
            )
        except Exception as exc:
            log(idx, f"1A CFG=2xUSP=2 note: {exc}")

        try:
            run_host_sweep(
                idx,
                bucket=args.gcs_bucket,
                gcs_out_prefix=gcs_prefix,
                config_meta={
                    "name": "mh3_bf16_u2_2chips_sdpa",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "BF16",
                    "chips": 2,
                    "num_gpus": 2,
                    "ulysses_degree": 2,
                    "attn_mode": "2-Chip USP=2 + XLA SDPA + AdaLN-online",
                },
                server_spec={
                    "config_name": "mh3_bf16_u2_2chips_sdpa",
                    "model_path": base_dir,
                    "port": port,
                    "num_gpus": 2,
                    "ulysses_degree": 2,
                    "quantization": None,
                    "v5p_splash": False,
                    "adaln_online": True,
                },
                workloads=cfg2_workloads[:3],
                keyframes=keyframes,
                existing_results=pod_results,
            )
        except Exception as exc:
            log(idx, f"1B 2-chip USP=2 note: {exc}")

    elif idx == 2:
        base_dir = stage_single_model_in_shm(idx, args.gcs_bucket, "minimax_h3")
        ablation_workloads = [
            {"label": "768p 16:9 15s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 15s i2v (25-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 5s i2v (4-step)", "task": "fl2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 5, "concurrency": 1, "num_requests": 2},
            {"label": "768p 16:9 5s t2va (25-step)", "task": "t2va", "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0, "num_inference_steps": 26, "concurrency": 1, "num_requests": 2},
        ]
        # 2A: Online AdaLN + XLA Fused SDPA (ablation test; pruned curve checkpoint embeds polynomial)
        try:
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
        except Exception as exc:
            log(idx, f"2A AdaLN-online ablation note: {exc} (pruned curve checkpoint embeds AdaLN polynomial)")

        # 2A-Standard: Standard XLA Fused SDPA on ablation workloads
        try:
            run_host_sweep(
                idx,
                bucket=args.gcs_bucket,
                gcs_out_prefix=gcs_prefix,
                config_meta={
                    "name": "mh3_bf16_u4_sdpa_ablation",
                    "model_short": "MiniMax-H3 (FL2VA)",
                    "is_fasth3": False,
                    "precision": "BF16",
                    "chips": 4,
                    "num_gpus": 4,
                    "ulysses_degree": 4,
                    "attn_mode": "XLA Fused SDPA",
                },
                server_spec={
                    "config_name": "mh3_bf16_u4_sdpa_ablation",
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
        except Exception as exc:
            log(idx, f"2A-Standard ablation note: {exc}")

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
                    "adaln_online": True,
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
                    "adaln_online": True,
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
                        "adaln_online": True,
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
