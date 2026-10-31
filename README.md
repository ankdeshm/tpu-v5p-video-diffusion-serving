# Cloud TPU `v5p` (`2x2x1` Single-Host, `2x2x4` `v5p-16` & `2x4x4` `v5p-32` Slice) Video+Audio Diffusion Serving (`MiniMax-H3`, `LTX-2.5`, `Wan 2.2 A14B`)

[![Benchmark Report](https://img.shields.io/badge/Benchmark-Report-blue)](V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md)
[![TPU Architecture](https://img.shields.io/badge/TPU-v5p--16%20%7C%20v5p--32-green)](#)
[![Engines](https://img.shields.io/badge/Engines-vllm--omni%20%7C%20sglang--torchtpu-orange)](#)

Reproducible serving infrastructure, containers, Kubernetes Indexed Job manifests, and multi-configuration benchmark harnesses for **multimodal video + stereo audio diffusion models** on **Cloud TPU `v5p`** (`95 GiB` HBM2e per chip / `380 GiB` HBM2e per 4-chip host, `459` BF16 TFLOP/s per chip) using **`sglang-torchtpu`** (`TpuOOTPlatform`, `tpu_splash_attn` / `torch_sdpa` + Ulysses/CFG Sequence Parallelism) and **`vllm-torchtpu` (`vLLM-Omni` v0.32.0)**:

1. **Priority #1 — [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) (`FL2VA` Joint Video+Audio `t2va` & `fl2va` I2V) & [`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) on `v5p-16` (`2x2x4`, 16 chips)**
   - **Target Specs:** `i2v 768P 15s` (`fl2va`, `1344×768`, `15.0s` @ `24 fps` = `360` frames + `32 kHz` stereo AAC audio, `4`-step distilled / `12`-step / `25`-step / `49`-step) + `5s` (`124f`), `10s` (`244f`), `9:16` Portrait (`768×1344`), and `512p` (`896×512`).
   - **Configurations:** Single-host `4x TPU v5p` chips (`380 GiB` HBM2e, `Ulysses=4`, `--encoder-parallel fold`, `--text-encoder-cpu-offload`) & `2x TPU v5p` chips (`Ulysses=2`), `BF16` vs Online `FP8` (`--quantization fp8`), `tpu_splash_attn` vs `torch_sdpa`, Precomputed vs Online `AdaLN`, and `16`-chip `v5p-16` `DP=4 × Ulysses=4` 30-RPM slice sizing (`c=4, 8`).
2. **Priority #2 — [`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) (`22B` Asymmetric Dual-Stream Video+Audio DiT + `2×` Diffusion Decoder Upscaler) on `v5p-32` (`2x4x4`, 32 chips)**
   - **Target Specs:** Standard `8`-step distilled (`960×544` & `1280×704` `720p` across `5s` `121f`, `10s` `241f`, `20s` `481f`), `20`-step & `40`-step dev schedules, I2VA, and **Two-Stage `1080p` (`1920×1088`, `use_diffusion_decoder=True`) `20s` (`481` frames @ `24 fps`, `bs=1`) Kling-V4 Proxy**.
   - **Configurations:** `sglang-torchtpu` (`BF16` vs `FP8`, `Ulysses=4` vs `CFG=2 × Ulysses=2`, `tpu_splash_attn` vs `torch_sdpa`) and `vllm-torchtpu` (`vLLM-Omni` `--omni --usp 4`), plus `32`-chip `v5p-32` `DP=8 × Ulysses=4` concurrency sweep (`c=8, 16`) & 30-RPM slice sizing.
3. **Priority #3 — [`Wan-AI/Wan2.2-I2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) & [`Wan-AI/Wan2.2-T2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (`28.6B` Total / `14.3B` Active Two-Stage High/Low-Noise MoE DiT) on `v5p-32` (`2x4x4`, 32 chips)**
   - **Target Specs:** `480p` (`832×480`, `81f` `5s` @ `16 fps`), `720p` (`1280×720`, `81f` `5s` & `161f` `10s`), `720p` `9:16` Portrait (`720×1280`, `81f`), across `4`-step fast, `20`-step, and `40`-step schedules.
   - **Configurations:** `sglang-torchtpu` (`BF16` & `FP8`, `Ulysses=4` & `CFG=2 × Ulysses=2`) and `vllm-torchtpu` (`vLLM-Omni` `--omni --usp 4`), plus `32`-chip `v5p-32` `DP=8 × Ulysses=4` slice throughput & 30-RPM sizing.

---

## 1. Executive Summary & Benchmark Highlights

Full detailed data, single-host latency matrices, and scaling models are available in the [Comprehensive Benchmark Report](V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md).

- **Real-Time Video Generation on TPU `v5p`:**
  - `LTX-2.5` on `vllm-omni` achieves **`4.62s` warm median E2E** for a 5-second `960x544` 121-frame video with stereo audio (`1.07x` real-time generation speed) on a single 4-chip `v5p` host (`12.97 RPM`).
  - At `720p` (`1280x704`), warm median latency is **`7.38s`** (`8.13 RPM` per host, `0.68x` real-time).
- **Wan 2.2 Serving Robustness:**
  - `Wan2.2-T2V-A14B` on `vllm-omni` achieves **`27.28s`** (`2.27 RPM`) on 480p 4-step fast, and **`44.66s`** (`1.34 RPM`) on 720p 4-step fast.
  - Deep 20-step iterations run in **`168.99s`** (`0.36 RPM`) on 480p and **`201.44s`** (`0.30 RPM`) on 720p.
- **Engine Comparison (`vllm-omni` vs `sglang-torchtpu`):**
  - **Cold Compilation & Boot:** `vllm-omni` demonstrates faster server boot (105-195s) and shorter XLA graph compilation for few-step distilled workloads (e.g., LTX-2.5 8-step compiled in ~101s on vllm-omni vs ~300s on sglang).
  - **Iteration Scalability:** `sglang-torchtpu` exhibits lower per-step execution overhead on deep iterative trajectories (20-40 steps) with Ulysses Sequence Parallelism.
- **Cross-Generation Comparison (`v7x` vs `v5p`):**
  - TPU `v7x-16` provides `192 GiB` HBM3e/chip with integrated XLA spatial attention, enabling resident serving of massive 33B DiT + multi-encoder pipelines without host RAM offloading.
  - TPU `v5p` (`95 GiB` HBM2e/chip, 32 GiB usable per TensorCore) requires folded encoder parallelism (`--encoder-parallel fold`) and selective text-encoder CPU offload to prevent device memory exhaustion on 30B+ DiT architectures.

---

## 2. 30 RPM Target Sizing Matrix (Host & Chip Counts)

To serve a production SLA of **30 Requests Per Minute (RPM)**, the required cluster infrastructure scales based on single-host throughput as follows:

| Model & Workload | Serving Engine | Parallelism | Warm Median Latency | Single Host (4 Chips) RPM | Hosts Needed for 30 RPM | Total Chips (`v5p`) | Slices (`v5p-16` / `v5p-32`) |
|---|---|---|---|---|---|---|---|
| **LTX-2.5 544p 5s 121f (8-step)** | `vllm-omni` | `USP=4` | **4.62s** | **12.97** | **3 hosts** | **12 chips** | 1× `v5p-16` |
| **LTX-2.5 720p 5s 121f (8-step)** | `vllm-omni` | `USP=4` | **7.38s** | **8.13** | **4 hosts** | **16 chips** | 1× `v5p-16` |
| **LTX-2.5 544p 10s 241f (8-step)** | `vllm-omni` | `USP=4` | **9.12s** | **6.58** | **5 hosts** | **20 chips** | 1× `v5p-32` |
| **LTX-2.5 720p 10s 241f (8-step)** | `vllm-omni` | `USP=4` | **14.55s** | **4.12** | **8 hosts** | **32 chips** | 1× `v5p-32` |
| **Wan2.2-T2V 480p 5s (4-step fast)** | `vllm-omni` | `USP=4` | **27.28s** | **2.27** | **14 hosts** | **56 chips** | 2× `v5p-32` |
| **Wan2.2-T2V 720p 5s (4-step fast)** | `vllm-omni` | `USP=4` | **44.66s** | **1.34** | **23 hosts** | **92 chips** | 3× `v5p-32` |
| **Wan2.2-T2V 480p 5s (20-step)** | `vllm-omni` | `USP=4` | **169.00s** | **0.36** | **84 hosts** | **336 chips** | 11× `v5p-32` |
| **Wan2.2-T2V 720p 5s (20-step)** | `vllm-omni` | `USP=4` | **201.44s** | **0.30** | **100 hosts** | **400 chips** | 13× `v5p-32` |
| **FastH3 768p 5s 124f (4-step VSA)** | `sglang` | `Ulysses=4` | **48.20s** | **1.24** | **25 hosts** | **100 chips** | 4× `v5p-32` (or 7× `v5p-16`) |
| **MiniMax-H3 768p 15s (25-step)** | `sglang` | `Ulysses=4` | **268.40s** | **0.22** | **137 hosts** | **548 chips** | 18× `v5p-32` (or 35× `v5p-16`) |

---

## 3. Key TPU `v5p` Serving Architecture & Optimizations

| Feature | Implementation | Impact |
|---|---|---|
| **Single-Host `2x2x1` Isolation on Multi-Host `2x2x4` / `2x4x4` Slices** | `LD_PRELOAD=/workspace/metadata_hook.so` + local `v5p`-aware `mock_metadata.py` (`TPU_ACCELERATOR_TYPE=v5p-8`, `CHIPS_PER_HOST_BOUNDS=2,2,1`) | Runs **4 independent `4`-chip (`380 GiB` HBM2e) servers** on `v5p-16` (`DP=4 × Ulysses=4`) and **8 independent `4`-chip servers** on `v5p-32` (`DP=8 × Ulysses=4`) with zero cross-host ICI barriers |
| **5-Point `MiniMax-H3` `sglang-torchtpu` TPU Compatibility Patchset** | TPU `ProcessGroupTPU` AdaLN device placement, non-CUDA VDN attention branch execution, CPU `get_rope_index` + `int32` token casting, static contiguous slice splice in `Qwen3-VL` deepstack/visual tokens, and `tpu` denoising stage enablement | Unlocks full `MiniMax-H3` (`FL2VA`: `t2va` & `fl2va` `i2v`) and `FastH3` 4-step VSA joint video+audio serving on `torch_tpu` |
| **`vllm-omni` LTX-2.5 Distilled Routing & Auto-Padding** | ConvVAE routing via `_ltx2_use_diffusion_decoder -> False` + `auto_pad=True` in `SequenceParallelInput` | Prevents token count divisibility crashes on irregular resolutions (`960x544`, `1280x704`) and maps directly to distilled transformer weights |
| **PyAV AAC Audio Frame Resampling** | `av.AudioResampler` chunking to 1024-sample packets before `avcodec_send_frame()` | Resolves PyAV `EINVAL (22)` crash during AAC stereo audio muxing in `vllm-omni` |
| **`95 GiB`/chip HBM Budget Management (`v5p` Host = `380 GiB` HBM)** | `--encoder-parallel fold` (`4`-way sharded `Qwen3-VL` `32B`) + `--text-encoder-cpu-offload` (or Online `FP8` `--quantization fp8`) | Comfortably fits `66.3 GB` `MiniMax-H3` DiT + `Qwen3-VL` + `3D Video VAE` + `Audio VAE` inside `380 GiB` single-host HBM |
| **TPU Seeded Generator Deadlock Prevention** | Passes `"generator_device": "cpu"` in `/v1/videos` requests | Avoids GIL deadlock in blocking host-to-device RNG seed copy on `torch_tpu` |
| **GCSFuse + `/dev/shm` Single-Active-Model Staging** | Streams active checkpoint shards into `/dev/shm` and purges between model families | Prevents `448 GiB` `ct5p-hightpu-4t` host RAM exhaustion while achieving multi-GB/s model load |

---

## 4. Repository Structure

```text
├── README.md                                          # This document
├── V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md             # Comprehensive benchmark report & technical analysis
├── docker/
│   ├── Dockerfile.sglang_tpu                          # TorchTPU 0.2.0 + sglang-torchtpu + varlen patch + ffmpeg
│   ├── prep_sglang_tpu.py                             # ABI constraint lock + TOML-verified sglang-torchtpu setup
│   ├── Dockerfile.vllm_omni                           # TorchTPU 0.2.0 + vllm-torchtpu v0.32.0 + vLLM-Omni (CPU/TPU target)
│   └── prep_vllm_omni.py                              # ABI constraint lock for vLLM-Omni on TPU
├── scripts/
│   ├── stage_all_video_models_gcs.py                  # Parallel HF Hub -> regional GCS stager for MiniMax-H3, FastH3, LTX-2.5, Wan 2.2
│   ├── run_v5p_minimax_h3_serving.py                  # v5p-16: MiniMax-H3 (FL2VA i2v + t2va) & FastH3 4-step VSA benchmark suite
│   ├── run_v5p_ltx25_wan22_omni.py                    # v5p-32: LTX-2.5 & Wan 2.2 serving harness for vllm-omni
│   └── run_v5p_ltx25_wan22_sglang.py                  # v5p-32: LTX-2.5 & Wan 2.2 serving harness for sglang-torchtpu
├── k8s/
│   ├── v5p-16-spot-minimax-h3-serving.yaml            # Indexed Job (4x4=16 v5p chips) on ank-tpu-v5p-16-spot
│   ├── v5p-32-spot-omni-serving.yaml                  # Indexed Job on ank-tpu-v5p-32-spot for vllm-omni comparative serving
│   └── v5p-32-spot-sglang-serving.yaml                # Indexed Job on ank-tpu-v5p-32-spot for sglang-torchtpu
└── results/
    ├── v5p16_spot_minimax_h3_serve_p1/                # MiniMax-H3 / FastH3 pod and node execution results
    ├── v5p32_spot_omni_serve_p23/                     # vLLM-Omni LTX-2.5 & Wan 2.2 pod and node results
    └── v5p32_spot_sglang_serve_p23/                   # SGLang LTX-2.5 & Wan 2.2 pod and node results
```

---

## 5. Quickstart

```bash
# 1. Upload serving benchmark harnesses to regional GCS
CLOUDSDK_METRICS_ENVIRONMENT=datacloud.antigravity gcloud storage cp scripts/run_v5p_*.py gs://ank-usc1-bucket-314837540096/v5p-video-serving-scripts/

# 2. Run MiniMax-H3 (FL2VA i2v + t2va) & FastH3 on ank-tpu-v5p-16-spot (4 hosts = 16 v5p chips)
CLOUDSDK_METRICS_ENVIRONMENT=datacloud.antigravity kubectl --context=gke_diesel-patrol-382622_us-central1_ank-tpu-v5p-usc1 apply -f k8s/v5p-16-spot-minimax-h3-serving.yaml

# 3. Run LTX-2.5 & Wan 2.2 comparative serving on ank-tpu-v5p-32-spot (8 hosts = 32 v5p chips)
CLOUDSDK_METRICS_ENVIRONMENT=datacloud.antigravity kubectl --context=gke_diesel-patrol-382622_us-central1_ank-tpu-v5p-usc1 apply -f k8s/v5p-32-spot-sglang-serving.yaml
CLOUDSDK_METRICS_ENVIRONMENT=datacloud.antigravity kubectl --context=gke_diesel-patrol-382622_us-central1_ank-tpu-v5p-usc1 apply -f k8s/v5p-32-spot-omni-serving.yaml
```
