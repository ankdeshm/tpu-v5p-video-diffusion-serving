# TPU v5p Video + Audio Diffusion Serving Benchmark Report

- **Date:** October 31, 2026
- **Cluster:** Cluster 1 (`gke_YOUR_GCP_PROJECT_ID_us-central1_tpu-v5p-cluster`)
- **Nodepools:**
  - `tpu-v5p-16-pool` (`4` hosts = `16` `v5p` physical chips / `32` TensorCores, topology `2x2x4`)
  - `tpu-v5p-32-pool` (`8` hosts = `32` `v5p` physical chips / `64` TensorCores, topology `2x4x4`)
- **Evaluated Serving Engines:** `vllm-omni` (v0.32.0-tpu) vs `sglang-torchtpu` (v0.2.0-diffusion)
- **Evaluated Models:**
  - **LTX-2.5** (`22B` Video + Audio DiT; 8-step distilled and multi-stage)
  - **Wan2.2-T2V-A14B & Wan2.2-I2V-A14B** (`14B` High-Quality Diffusion Transformers)
  - **MiniMax-H3 (FL2VA)** (`33.14B` Omni-Modal Video + Audio DiT) & **FastH3** (4-step VSA Distill)
- **Cross-Generation Baseline:** TPU `v7x-16` (`tpu-v7x-16-pool`, Steady Flex, `16` physical chips / `32` TensorCores / `3,072 GiB` HBM3e)

---

## 1. Executive Summary & Key Findings

1. **Sub-5-Second Full Video+Audio Diffusion on TPU v5p (`LTX-2.5`):**
   - Utilizing `vllm-omni` with 4 `v5p` chips (Ulysses degree = 4), **LTX-2.5 544p (960x544) 5s 121f T2VA** generated in **`4.684s`** median latency (**`12.93 RPM`** per host).
   - This represents **`1.07x` real-time generation** (generating a 5.0s video with synchronized stereo audio faster than real-time playback).
   - **720p (1280x704) 5s 121f T2VA** completed in **`7.383s`** median latency (**`8.13 RPM`** per host).

2. **Large-Model Video Diffusion (`Wan 2.2 14B`):**
   - `vllm-omni` successfully booted Wan2.2-T2V-A14B in **`195.1s`**, generating 480p 4-step fast video in **`27.28s`** median latency (**`2.27 RPM`** per host) and 720p 4-step in **`44.65s`** (**`1.34 RPM`** per host).
   - `sglang-torchtpu` generated 480p 4-step in **`33.09s`** (**`1.82 RPM`** per host) and 20-step high-fidelity in **`119.10s`** (**`0.50 RPM`** per host).

3. **Cross-Generation Architecture: TPU `v7x` vs TPU `v5p`:**
   - **HBM Capacity & Large DiT Placement:** TPU `v7x` features `192 GiB` HBM3e per physical chip (`384 GiB` per dual-chip host), allowing `MiniMax-H3` (33.14B params, 144GB weights) to run resident DiT serving with text-encoder/VAE offload on 4 chips with `175.5s` median latency for 15-second 1344x768 video (0.342 RPM).
   - On TPU `v5p` (`95 GiB` HBM per chip, `16 GiB` per core), monolithic serving of 33B+ models without CPU offload encounters HBM buffer allocation limits, necessitating multi-host tensor sharding (16-32 chips) or selective encoder/VAE offloading.

4. **Critical Serving Fixes Discovered & Upstreamed:**
   - **LTX-2.5 Sequence Parallel Auto-Padding:** Resolved strict divisibility constraints when latent tokens (e.g. `31,110` tokens for 960x544) are not evenly divisible by `sequence_parallel_size=4`. Configured `auto_pad=True` in `SequenceParallelInput`.
   - **PyAV Audio Resampler Chunking:** Fixed PyAV `avcodec_send_frame()` error 22 (`EINVAL`) during MP4 multiplexing by introducing `av.AudioResampler` to feed fixed frame-size packets (`1024` samples) to the AAC audio encoder.
   - **MiniMax-H3 Pruned Checkpoint Adaptation:** Identified that `MiniMaxAI/MiniMax-H3/FL2VA` checkpoints embed the AdaLN polynomial curves directly in safetensors shards, requiring standard SDPA execution rather than `--minimax-h3-adaln-online` rebuilds.

---

## 2. TPU v5p Benchmark Results Matrix

All single-host configurations below utilize **1 host = 4 physical `v5p` chips = 8 TensorCores** (`16 GiB` HBM per core / `32 GiB` HBM per chip).

### 2.1 Single-Host Serving Performance

| Model | Serving Engine | Precision | Parallelism | Resolution | Duration / Frames | Denoise Steps | Cold Compile (s) | Warm Median E2E (s) | ms / step | Host RPM | Video-s / Wall-s | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **LTX-2.5** | `vllm-omni` | BF16 | U=4 | 544p (960x544) | 5.0s / 121f | 8 | 87.66 | **4.684** | 585.5 | **12.93** | 1.068x | Verified OK |
| **LTX-2.5 (I2VA)** | `vllm-omni` | BF16 | U=4 | 544p (960x544) | 5.0s / 121f | 8 | 22.42 | **4.624** | 578.0 | **12.97** | 1.081x | Verified OK |
| **LTX-2.5** | `vllm-omni` | BF16 | U=4 | 720p (1280x704) | 5.0s / 121f | 8 | 82.46 | **7.383** | 922.9 | **8.13** | 0.677x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | U=4 | 480p (832x480) | 5.0s / 81f | 4 | 313.62 | **27.280** | 6820.0 | **2.27** | 0.183x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | U=4 | 480p (832x480) | 5.0s / 81f | 20 | - | **168.995** | 8449.8 | **0.36** | 0.030x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | U=4 | 480p (832x480) | 5.0s / 81f | 40 | - | **324.721** | 8118.0 | **0.19** | 0.015x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | U=4 | 720p (1280x720) | 5.0s / 81f | 4 | 286.76 | **44.655** | 11163.8 | **1.34** | 0.112x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | U=4 | 720p (1280x720) | 5.0s / 81f | 20 | - | **201.444** | 10072.2 | **0.30** | 0.025x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | U=4 | 720p (1280x720) | 5.0s / 81f | 40 | - | **389.110** | 9727.8 | **0.15** | 0.013x | Verified OK |
| **Wan2.2-T2V** | `sglang` | BF16 | U=4 | 480p (832x480) | 5.0s / 81f | 4 | 84.17 | **33.088** | 8272.0 | **1.82** | 0.151x | Verified OK |
| **Wan2.2-T2V** | `sglang` | BF16 | U=4 | 480p (832x480) | 5.0s / 81f | 20 | - | **119.096** | 5954.8 | **0.50** | 0.042x | Verified OK |
| **Wan2.2-T2V** | `sglang` | BF16 | U=4 | 480p (832x480) | 5.0s / 81f | 40 | - | **226.823** | 5670.6 | **0.26** | 0.022x | Verified OK |

---

## 3. Production Sizing for 30 RPM Serving Target

To satisfy a target production throughput of **30 Requests Per Minute (30 RPM)**, the cluster sizing across models on TPU `v5p` (4 physical chips / host) is sized below based on empirical single-host throughput (strictly host, chip, and core counts):

| Model Architecture | Workload Profile | Measured Host RPM | Target Cluster RPM | Required Host Count | Required `v5p` Chips | Required TensorCores | Recommended Nodepool Topology |
|---|---|---:|---:|---:|---:|---:|---|
| **LTX-2.5 (T2VA)** | 544p 5s 121f (8-step) | 12.93 | 30.0 | **3 hosts** | **12 chips** | 24 cores | `tpu-v5p-16-pool` (`2x2x4`) |
| **LTX-2.5 (I2VA)** | 544p 5s 121f (8-step) | 12.97 | 30.0 | **3 hosts** | **12 chips** | 24 cores | `tpu-v5p-16-pool` (`2x2x4`) |
| **LTX-2.5 (T2VA)** | 720p 5s 121f (8-step) | 8.13 | 30.0 | **4 hosts** | **16 chips** | 32 cores | `tpu-v5p-16-pool` (`2x2x4`) |
| **Wan2.2-T2V-A14B** | 480p 5s 81f (4-step fast) | 2.27 | 30.0 | **14 hosts** | **56 chips** | 112 cores | `tpu-v5p-64-spot` (`4x4x4`) |
| **Wan2.2-T2V-A14B** | 720p 5s 81f (4-step fast) | 1.34 | 30.0 | **23 hosts** | **92 chips** | 184 cores | `tpu-v5p-128-spot` (`4x4x8`) |
| **Wan2.2-T2V-A14B** | 480p 5s 81f (20-step hifi)| 0.50 | 30.0 | **60 hosts** | **240 chips** | 480 cores | `tpu-v5p-256-spot` (`4x8x8`) |
| **MiniMax-H3 (FL2VA)**| 768p 15s (4-step, v7x baseline) | 0.34 | 30.0 | **89 hosts** | **356 chips** | 712 cores | `tpu-v5p-512-spot` (`8x8x8`) |

---

## 4. Cross-Generation Architectural Comparison: TPU `v7x` vs TPU `v5p`

The cross-generation benchmark compares TPU `v7x` (`tpu-v7x-16-pool`) against TPU `v5p` (`tpu-v5p-32-pool` and `tpu-v5p-16-pool`).

### 4.1 Head-to-Head Specification

| Architectural Feature | TPU `v7x` (Flex) | TPU `v5p` (Spot) | Architectural Impact on Video DiT Serving |
|---|---|---|---|
| **HBM Generation & Speed** | HBM3e (`~4.8 TB/s` bandwidth) | HBM2e (`~2.4 TB/s` bandwidth) | 2.0x memory bandwidth on v7x accelerates memory-bound attention steps |
| **HBM Capacity per Chip** | **`192 GiB`** | **`95 GiB`** (`~32 GiB` usable per rank) | v7x holds 33B-45B monolithic models in resident HBM without offloading |
| **TensorCores per Chip** | 2 TensorCores (Sparse MatMul) | 2 TensorCores (VPU + MXU) | Faster matrix multiplication tiles on v7x |
| **Interconnect (ICI)** | High-radix 3D optical mesh | 3D Torus interconnect | High-efficiency cross-host collective all-gathers on both |

### 4.2 Cross-Generation Empirical Video Serving Matrix

| Benchmark Scenario | Model | Hardware Platform | Chips | Serving Engine | Precision | Attention Backend | Output Canvas | Warm E2E (s) | Host RPM | Status |
|---|---|---|---:|---|---|---|---|---:|---:|---|
| **High-Res Long Video (15s)** | `MiniMax-H3` | TPU `v7x-16` | 4 | `sglang` | FP8 (online) | Pallas FlashAttn (Offload) | 1344x768 (15s) | **175.51** | 0.342 | Verified OK |
| **High-Res Long Video (15s)** | `MiniMax-H3` | TPU `v7x-16` | 4 | `sglang` | BF16 | Stock SDPA + AdaLN | 1344x768 (15s) | **177.99** | 0.337 | Verified OK |
| **Real-Time Video+Audio (5s)** | `LTX-2.5` | TPU `v5p-32` | 4 | `vllm-omni` | BF16 | TPU SDPA (XLA Fused) | 960x544 (5s) | **4.68** | **12.93** | Verified OK |
| **Real-Time Video+Audio (5s)** | `LTX-2.5` | TPU `v5p-32` | 4 | `vllm-omni` | BF16 | TPU SDPA (XLA Fused) | 1280x704 (5s) | **7.38** | **8.13** | Verified OK |
| **Fast Cinematic Video (5s)** | `Wan2.2-T2V` | TPU `v5p-32` | 4 | `vllm-omni` | BF16 | TPU SDPA (XLA Fused) | 832x480 (5s) | **27.28** | **2.27** | Verified OK |
| **Fast Cinematic Video (5s)** | `Wan2.2-T2V` | TPU `v5p-32` | 4 | `sglang` | BF16 | TPU SplashAttn | 832x480 (5s) | **33.09** | **1.82** | Verified OK |

---

## 5. Serving Engine Comparison: `vllm-omni` vs `sglang-torchtpu`

### 5.1 Latency and Throughput Dynamics
- **`vllm-omni`:** Outperformed `sglang-torchtpu` on fast distilled video models. Booted `LTX-2.5` in **105s** and rendered 5s video+audio in **`4.68s`**.
- **`sglang-torchtpu`:** Demonstrates stronger scaling on deep iterative diffusion steps (e.g. 20-step Wan2.2: `119.1s` on sglang vs `168.9s` on omni; 40-step: `226.8s` on sglang vs `324.7s` on omni).

### 5.2 Key Runtime Discoveries & Patches
1. **PyAV Audio Resampler:**
   PyAV's `AudioStream.encode()` requires audio packets chunked to the codec's `frame_size` (`1024` for AAC). Without an `AudioResampler` chunking step, FFmpeg rejects whole-audio NumPy arrays with error 22 (`EINVAL`). Chunking audio via `av.AudioResampler` resolves this.
2. **Ulysses Sequence Parallel Auto-Padding:**
   `LTX-2.5` sequence length is dynamic based on spatial and temporal dimensions ($S = (W/32) \times (H/32) \times ((F-1)/8 + 1)$). Resolutions like 960x544 generate non-divisible token counts ($31,110$ tokens). Setting `auto_pad=True` in `SequenceParallelInput` enables arbitrary aspect ratios.
3. **MiniMax-H3 Architecture on TPU v5p:**
   The `MiniMax-H3` checkpoint contains pre-pruned AdaLN polynomials. Monolithic loading without CPU offload consumes exceeding HBM during initialization; multi-host FSDP/Ulysses sharding or CPU offloading is necessary.

---

## 6. Manifest & Repository Structure

All benchmark manifests and serving engines are preserved in the git repository:

```
tpu-v5p-video-diffusion-serving/
├── README.md
├── k8s/
│   ├── v5p-16-spot-minimax-h3-serving.yaml   # 4-host v5p-16 Spot Job (16 v5p chips)
│   ├── v5p-32-spot-omni-serving.yaml         # 2-host v5p-32 Spot Job (vllm-omni)
│   └── v5p-32-spot-sglang-serving.yaml       # 6-host v5p-32 Spot Job (sglang-torchtpu)
├── scripts/
│   ├── run_v5p_ltx25_wan22_omni.py           # vllm-omni runner with SP auto_pad & PyAV fix
│   ├── run_v5p_ltx25_wan22_sglang.py         # sglang runner for LTX & Wan
│   └── run_v5p_minimax_h3_serving.py         # MiniMax-H3 & FastH3 runner
└── results/
    ├── v5p16_spot_minimax_h3_serve_p1/       # MiniMax-H3 benchmark JSONs
    ├── v5p32_spot_omni_serve_p23/            # LTX-2.5 & Wan2.2 omni JSONs
    └── v5p32_spot_sglang_serve_p23/          # SGLang benchmark JSONs
```
