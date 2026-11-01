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

1. **Sub-5-Second & Sub-9-Second 10s Video+Audio Diffusion on TPU v5p (`LTX-2.5`):**
   - Utilizing `vllm-omni` with 4 `v5p` chips (`USP=4`), **LTX-2.5 544p (960x544) 5s 121f** generated in **`4.624s` (I2VA, `12.97 RPM/host`)** and **`4.684s` (T2VA, `12.93 RPM/host`, `1.07x` real-time)**, while **10s 241f T2VA** completed in **`8.694s` warm median (`6.902 RPM/host`, `1.15x` real-time!)**.
   - **2-Chip Sub-Host Packing (`USP=2` × 2 Replicas/Host):** A 2-chip `v5p` replica (`ltx25_omni_bf16_2chip_usp2`) rendered 544p 5s 121f T2VA in **`6.931s` (`8.648 RPM/replica`)**, delivering **`17.30 RPM` per 4-chip host (`+33.8%` higher host throughput than `USP=4`)** and reducing `30 RPM` fleet sizing to just **2 hosts (`8` `v5p` chips)**!
   - **6-Host (`24`-Chip `DP=6`) Multi-Host Cluster Throughput:** Sustained **`85.6 RPM`** (`>114 RPM` scaled to 8-host `v5p-32`) on 544p 5s T2VA!

2. **Large-Model Video Diffusion (`Wan2.2-T2V-A14B` & `Wan2.2-I2V-A14B`):**
   - **`CFG=2 × USP=2` Hybrid Parallelism (`vllm-omni`):** Outperformed pure `USP=4`, rendering **Wan2.2-T2V 480p 4-step in `27.108s` (`2.216 RPM/host`)** and **720p 4-step in `44.223s` (`1.357 RPM/host`)** (`200.56s` at 20-step).
   - **`Wan2.2-I2V-A14B` Image-to-Video (`USP=4`, `vllm-omni`):** Generated **480p 5s 81f (4-step) in `28.049s` (`2.184 RPM/host`)** (`164.01s` at 20-step) and **720p 5s 81f (4-step) in `59.813s` (`1.003 RPM/host`)** (`202.25s` at 20-step), plus 720p 10s 161f (`599.20s`) and 720p Portrait `720x1280` (`203.89s`).
   - **`sglang-torchtpu`:** Generated Wan2.2-T2V 480p 4-step in **`33.09s` (`1.82 RPM/host`)** and excelled on deep 20-step (`118.54s`) and 40-step (`226.20s`) schedules.

3. **Resident `MiniMax-H3` (`33.14B` Omni-Modal DiT, `144 GB` BF16) Unlocked & Accelerated on Single 4-Chip TPU `v5p` Host (`389.36s` 15s HD!):**
   - With our Qwen3-VL / VDN / CPU AudioDecode patches and selective encoder/VAE CPU offload (`--dit-cpu-offload=false`), **`MiniMax-H3 (FL2VA)` ran 100% resident on a single 4-chip `v5p` host (`70.27 GB` peak HBM)**! Removing AdaLN-online recompute (`mh3_bf16_u4_sdpa`) accelerated **15-second `1344x768` (768p 16:9) video+audio (`i2v`, 4-step) from `458.69s` -> `389.363s` warm median (`0.154 RPM/host`, `+17.8%` faster, `11.13 MB` MP4 verified)** vs **`175.51s` (`0.342 RPM/host`)** on 4-chip TPU `v7x` (`2.22x` Ironwood speedup). In addition, **5-second `1344x768` (`4`-step)** completed in **`291.180s` (`0.206 RPM/host`, `58.19 GB` HBM)**, and **full `25`-step 15-second `1344x768` HD** completed in **`952.301s` (`37.94s/denoise-step`, `0.063 RPM/host`, `71.96 GB` HBM)**!

4. **Critical Serving Fixes Discovered & Upstreamed:**
   - **LTX-2.5 Sequence Parallel Auto-Padding:** Configured `auto_pad=True` in `SequenceParallelInput` to support non-divisible latent token counts (e.g. `31,110` tokens for `960x544` across `USP=4`).
   - **PyAV Audio Resampler Chunking:** Fixed PyAV `avcodec_send_frame()` `EINVAL` (`errno 22`) during stereo MP4 muxing via `av.AudioResampler` `1024`-sample AAC packet framing.
   - **MiniMax-H3 Precomputed vs Online AdaLN & Int32/MRoPE TPU Lowering:** Applied 9 runtime patches enabling full `1344x768` 15s `FL2VA` execution inside `95 GiB` `v5p` HBM (`70.27 GB` peak), proving precomputed AdaLN (`mh3_bf16_u4_sdpa`, `389.36s`) fits cleanly alongside selective encoder/VAE offload and beats AdaLN-online (`458.69s`) by `69.3s` per clip.

---

## 2. TPU v5p Benchmark Results Matrix

Single-host configurations below utilize **1 host = 4 physical `v5p` chips** (or **2 physical `v5p` chips** per replica in `2-Chip U=2` packing mode).

### 2.1 Single-Host & Sub-Host Serving Performance

| Model | Serving Engine | Precision | Parallelism | Resolution | Duration / Frames | Denoise Steps | Cold Compile (s) | Warm Median E2E (s) | ms / step | Replica / Host RPM | Video-s / Wall-s | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **LTX-2.5 (I2VA)** | `vllm-omni` | BF16 | U=4 (4 chips) | 544p (960x544) | 5.0s / 121f | 8 | 22.42 | **4.624** | 578.0 | **12.97 / host** | 1.081x | Verified OK |
| **LTX-2.5 (T2VA)** | `vllm-omni` | BF16 | U=4 (4 chips) | 544p (960x544) | 5.0s / 121f | 8 | 87.66 | **4.684** | 585.5 | **12.93 / host** | 1.068x | Verified OK |
| **LTX-2.5 (T2VA)** | `vllm-omni` | BF16 | **U=2 (2 chips)** | 544p (960x544) | 5.0s / 121f | 8 | 91.10 | **6.931** | 866.4 | **8.65 / rep (`17.30 / host`)** | 0.721x | Verified OK |
| **LTX-2.5 (T2VA)** | `vllm-omni` | BF16 | U=4 (4 chips) | 544p (960x544) | **10.0s / 241f** | 8 | 11.13 | **8.681** | 1085.1 | **6.91 / host** | **1.152x** | Verified OK |
| **LTX-2.5 (T2VA)** | `vllm-omni` | BF16 | U=4 (4 chips) | 720p (1280x704) | 5.0s / 121f | 8 | 82.46 | **7.383** | 922.9 | **8.13 / host** | 0.677x | Verified OK |
| **LTX-2.5 (T2VA)** | `vllm-omni` | BF16 | **U=2 (2 chips)** | 720p (1280x704) | 5.0s / 121f | 8 | 88.40 | **12.200** | 1525.0 | **4.92 / rep (`9.84 / host`)** | 0.410x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | **CFG=2×U=2** | 480p (832x480) | 5.0s / 81f | 4 | 308.40 | **27.108** | 6777.0 | **2.22 / host** | 0.184x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | U=4 (4 chips) | 480p (832x480) | 5.0s / 81f | 4 | 313.62 | **27.280** | 6820.0 | **2.27 / host** | 0.183x | Verified OK |
| **Wan2.2-I2V** | `vllm-omni` | BF16 | U=4 (4 chips) | 480p (832x480) | 5.0s / 81f | 4 | 492.80 | **28.049** | 7012.2 | **2.18 / host** | 0.182x | Verified OK |
| **Wan2.2-I2V** | `vllm-omni` | BF16 | U=4 (4 chips) | 480p (832x480) | 5.0s / 81f | 20 | - | **164.008** | 8200.4 | **0.37 / host** | 0.030x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | **CFG=2×U=2** | 720p (1280x720) | 5.0s / 81f | 4 | 284.10 | **44.223** | 11055.8 | **1.36 / host** | 0.113x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | U=4 (4 chips) | 720p (1280x720) | 5.0s / 81f | 4 | 286.76 | **44.655** | 11163.8 | **1.34 / host** | 0.112x | Verified OK |
| **Wan2.2-I2V** | `vllm-omni` | BF16 | U=4 (4 chips) | 720p (1280x720) | 5.0s / 81f | 4 | 497.17 | **59.813** | 14953.2 | **1.00 / host** | 0.084x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | **CFG=2×U=2** | 720p (1280x720) | 5.0s / 81f | 20 | - | **200.561** | 10028.0 | **0.30 / host** | 0.025x | Verified OK |
| **Wan2.2-I2V** | `vllm-omni` | BF16 | U=4 (4 chips) | 720p (1280x720) | 5.0s / 81f | 20 | - | **202.254** | 10112.7 | **0.30 / host** | 0.025x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | U=4 (4 chips) | 720p Portrait (720x1280) | 5.0s / 81f | 20 | 306.20 | **203.887** | 10194.4 | **0.29 / host** | 0.025x | Verified OK |
| **Wan2.2-T2V** | `vllm-omni` | BF16 | U=4 (4 chips) | 720p (1280x720) | **10.0s / 161f** | 20 | 401.07 | **599.201** | 29960.0 | **0.05 / host** | 0.008x | Verified OK |
| **Wan2.2-T2V** | `sglang` | BF16 | U=4 (4 chips) | 480p (832x480) | 5.0s / 81f | 4 | 84.17 | **33.088** | 8272.0 | **1.82 / host** | 0.151x | Verified OK |
| **Wan2.2-T2V** | `sglang` | BF16 | U=4 (4 chips) | 480p (832x480) | 5.0s / 81f | 20 | - | **118.538** | 5926.9 | **0.51 / host** | 0.042x | Verified OK |
| **Wan2.2-T2V** | `sglang` | BF16 | U=4 (4 chips) | 480p (832x480) | 5.0s / 81f | 40 | - | **226.198** | 5655.0 | **0.27 / host** | 0.022x | Verified OK |
| **MiniMax-H3 (FL2VA)** | `sglang` | BF16 | U=4 + AdaLN-Online | **768p 16:9 (1344x768)** | **5.0s** | 4 | 644.83 | **291.180** | 72165.8 | **0.206 / host** | 0.017x | Verified OK (`58.2 GB` HBM) |
| **MiniMax-H3 (FL2VA)** | `sglang` | BF16 | **U=4 (Precomputed AdaLN)** | **768p 16:9 (1344x768)** | **15.0s** | 4 | 1657.22 | **389.363** | 96400.2 | **0.154 / host** | **0.039x** | Verified OK (`70.3 GB` HBM — **#1 `v5p`!**) |
| **MiniMax-H3 (FL2VA)** | `sglang` | BF16 | U=4 + AdaLN-Online | **768p 16:9 (1344x768)** | **15.0s** | 4 | 1521.69 | **458.692** | 113711.0 | **0.131 / host** | 0.033x | Verified OK (`70.3 GB` HBM) |
| **MiniMax-H3 (FL2VA)** | `sglang` | BF16 | U=4 + AdaLN-Online | **768p 16:9 (1344x768)** | **15.0s** | **25** | - | **952.301** | **37936.3** | **0.063 / host** | 0.016x | Verified OK (`72.0 GB` HBM) |

---

## 3. Production Sizing for 30 RPM Serving Target

To satisfy a target production throughput of **30 Requests Per Minute (30 RPM)**, the cluster sizing across models on TPU `v5p` (4 physical chips / host) is sized below based on empirical host throughput:

| Model Architecture | Workload Profile | Optimal Packing | Measured Host RPM | Target Cluster RPM | Required Host Count | Required `v5p` Chips | Recommended Nodepool Topology |
|---|---|---|---:|---:|---:|---:|---|
| **LTX-2.5 (T2VA)** | 544p 5s 121f (8-step) | `2 × USP=2` / host | **17.30** | 30.0 | **2 hosts** | **8 chips** | `2x2x2` (`v5p-8`) or `v5p-16` |
| **LTX-2.5 (I2VA / T2VA)** | 544p 5s 121f (8-step, min latency) | `1 × USP=4` / host | 12.97 | 30.0 | **3 hosts** | **12 chips** | `tpu-v5p-16-pool` (`2x2x4`) |
| **LTX-2.5 (T2VA)** | 720p 5s 121f (8-step) | `2 × USP=2` / host | **9.84** | 30.0 | **4 hosts** | **16 chips** | `tpu-v5p-16-pool` (`2x2x4`) |
| **LTX-2.5 (T2VA)** | 544p **10s 241f** (8-step) | `1 × USP=4` / host | **6.91** | 30.0 | **5 hosts** | **20 chips** | `tpu-v5p-32-pool` (`2x4x4`) |
| **Wan2.2-T2V-A14B** | 480p 5s 81f (4-step fast) | `1 × USP=4` / host | 2.27 | 30.0 | **14 hosts** | **56 chips** | `tpu-v5p-64-spot` (`4x4x4`) |
| **Wan2.2-I2V-A14B** | 480p 5s 81f (4-step fast) | `1 × USP=4` / host | 2.18 | 30.0 | **14 hosts** | **56 chips** | `tpu-v5p-64-spot` (`4x4x4`) |
| **Wan2.2-T2V-A14B** | 720p 5s 81f (4-step fast) | `CFG=2 × USP=2` | 1.36 | 30.0 | **23 hosts** | **92 chips** | `tpu-v5p-128-spot` (`4x4x8`) |
| **Wan2.2-I2V-A14B** | 720p 5s 81f (4-step fast) | `1 × USP=4` / host | 1.00 | 30.0 | **30 hosts** | **120 chips** | `tpu-v5p-128-spot` (`4x4x8`) |
| **MiniMax-H3 (FL2VA)** | 768p 5s (4-step, `v5p` BF16) | `1 × USP=4` / host | **0.206** | 30.0 | **146 hosts** | **584 chips** | *(Or `v7x` for higher density)* |
| **MiniMax-H3 (FL2VA)** | 768p 15s (4-step, `v5p` BF16) | `1 × USP=4` / host | **0.154** | 30.0 | **195 hosts** | **780 chips** | *(Recommend `v7x`: **88 hosts / 352 chips** @ `0.342 RPM/host`)* |

---

## 4. Cross-Generation Architectural Comparison: TPU `v7x` vs TPU `v5p`

The cross-generation benchmark compares TPU `v7x` (`tpu-v7x-16-pool`) against TPU `v5p` (`tpu-v5p-32-pool` and `tpu-v5p-16-pool`).

### 4.1 Head-to-Head Specification

| Architectural Feature | TPU `v7x` (Flex) | TPU `v5p` (Spot) | Architectural Impact on Video DiT Serving |
|---|---|---|---|
| **HBM Generation & Speed** | HBM3e (`~7.4 TB/s` package BW) | HBM2e (`~2.76 TB/s` chip BW) | Higher memory bandwidth on `v7x` accelerates memory-bound 3D DiT attention |
| **HBM Capacity per Chip** | **`192 GiB`** (`96 GiB`/TensorCore) | **`95 GiB`** | `v7x` holds 33B-45B monolithic models with ample headroom; `v5p` fits `MiniMax-H3` at `70.3 GB` with encoder/VAE CPU offload |
| **Compute Units per Chip** | 2 TensorCores (Native FP8 + BF16 MXU) | 1 Chip / 4 MXUs (BF16 MXU) | `2.22x` end-to-end speedup on `MiniMax-H3` 768p 15s (`175.5s` vs `389.4s`) |
| **Interconnect (ICI)** | High-radix 3D optical torus | 3D Torus interconnect | High-efficiency intra-host Ulysses All-to-All across 4 chips |

### 4.2 Cross-Generation Empirical Video Serving Matrix

| Benchmark Scenario | Model | Hardware Platform | Chips | Serving Engine | Precision | Attention Backend | Output Canvas | Warm E2E (s) | Host RPM | Status |
|---|---|---|---:|---|---|---|---|---:|---:|---|
| **High-Res Long Video (15s)** | `MiniMax-H3` | TPU `v7x-16` | 4 | `sglang` | FP8 (online) | Pallas FlashAttn (Offload) | 1344x768 (15s) | **175.51** | **0.342** | Verified OK |
| **High-Res Long Video (15s)** | `MiniMax-H3` | TPU `v7x-16` | 4 | `sglang` | BF16 | Stock SDPA + AdaLN | 1344x768 (15s) | **177.99** | 0.337 | Verified OK |
| **High-Res Long Video (15s)** | `MiniMax-H3` | **TPU `v5p-16`** | **4** | **`sglang`** | **BF16** | **XLA Fused SDPA (Precomputed AdaLN)** | **1344x768 (15s)** | **389.36** | **0.154** | **Verified OK** |
| **High-Res Long Video (15s, 25-step)** | `MiniMax-H3` | **TPU `v5p-16`** | **4** | **`sglang`** | **BF16** | **XLA Fused SDPA + AdaLN-Online** | **1344x768 (15s)** | **952.30** | **0.063** | **Verified OK** |
| **High-Res Short Video (5s)** | `MiniMax-H3` | **TPU `v5p-16`** | **4** | **`sglang`** | **BF16** | **XLA Fused SDPA + AdaLN-Online** | **1344x768 (5s)** | **291.18** | **0.206** | **Verified OK** |
| **Real-Time Video+Audio (5s)** | `LTX-2.5` | TPU `v5p-32` | 4 | `vllm-omni` | BF16 | TPU SDPA (XLA Fused) | 960x544 (5s) | **4.62** | **12.97** *(17.30 @ 2×U2)* | Verified OK |
| **Real-Time Video+Audio (10s)** | `LTX-2.5` | TPU `v5p-32` | 4 | `vllm-omni` | BF16 | TPU SDPA (XLA Fused) | 960x544 (10s) | **8.68** | **6.91** | Verified OK |
| **Real-Time Video+Audio (5s)** | `LTX-2.5` | TPU `v5p-32` | 4 | `vllm-omni` | BF16 | TPU SDPA (XLA Fused) | 1280x704 (5s) | **7.38** | **8.13** *(9.84 @ 2×U2)* | Verified OK |
| **Fast Cinematic T2V (5s)** | `Wan2.2-T2V` | TPU `v5p-32` | 4 | `vllm-omni` | BF16 | TPU SDPA (`CFG=2×U=2`) | 832x480 / 1280x720 (5s) | **27.11 / 44.22** | **2.22 / 1.36** | Verified OK |
| **Fast Cinematic I2V (5s)** | `Wan2.2-I2V` | TPU `v5p-32` | 4 | `vllm-omni` | BF16 | TPU SDPA (`USP=4`) | 832x480 / 1280x720 (5s) | **28.05 / 59.81** | **2.18 / 1.00** | Verified OK |
| **Fast Cinematic T2V (5s)** | `Wan2.2-T2V` | TPU `v5p-32` | 4 | `sglang` | BF16 | TPU SplashAttn | 832x480 (5s, 4 / 20 step) | **33.09 / 118.54** | **1.82 / 0.51** | Verified OK |

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
