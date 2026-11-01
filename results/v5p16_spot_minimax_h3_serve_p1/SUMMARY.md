# Priority #1 — MiniMax-H3 & FastH3 Video+Audio Serving on TPU v5p (`v5p16_spot_minimax_h3_serve_p1`)

- **Updated (UTC):** `2026-11-01T02:46:47.793116+00:00`
- **Cluster / Nodepool:** `tpu-v5p-cluster` / `tpu-v5p-16-pool` (`2x2x4`, `4` hosts × `4` physical `v5p` chips = `16` physical chips / `16` MegaCore devices / `1,520 GiB` HBM2e)
- **Serving Engine:** `sglang-torchtpu` (`TpuOOTPlatform`, XLA Fused SDPA vs TPU v5p Pallas SplashAttention + Ulysses Sequence Parallelism, 24 fps H.264 + 32 kHz stereo AAC muxed `.mp4`)
- **Registered Hosts:** `{"0": "v5p16-minimax-h3-p0-48l9q", "1": "v5p16-minimax-h3-p1-lgg6h", "2": "v5p16-minimax-h3-p2-jjmfb", "3": "v5p16-minimax-h3-p3-rmpzx"}`

## 1. Single-Host (`4x TPU v5p Chips` = `4` MegaCore Devices / `380 GiB` HBM2e) Configuration & Input-Distribution Matrix

| Config | Model | Prec. | Chips | Attn Mode | Task & Input Spec | Canvas | Steps | Conc. | Cold Compile (s) | Warm Med E2E (s) | ms/step | Host RPM (vids/min) | Video-s / Wall-s | OK |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 15s i2v (4-step)** (`fl2va`) | `-` | 4 | 1 | 0.114 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 15s t2va (4-step)** (`t2va`) | `-` | 4 | 1 | 0.004 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 15s i2v (12-step)** (`fl2va`) | `-` | 12 | 1 | - | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 15s i2v (25-step)** (`fl2va`) | `-` | 25 | 1 | - | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 15s i2v (49-step full)** (`fl2va`) | `-` | 49 | 1 | - | **-** | - | **0.00** | 0.00x | `0/1` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 5s i2v (4-step)** (`fl2va`) | `-` | 4 | 1 | 1800.367 | **-** | - | **0.00** | 0.00x | `0/3` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 5s t2va (25-step)** (`t2va`) | `-` | 25 | 1 | 1800.172 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 5s t2va (49-step full)** (`t2va`) | `-` | 49 | 1 | - | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 10s i2v (4-step)** (`fl2va`) | `-` | 4 | 1 | 1800.542 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 10s i2v (25-step)** (`fl2va`) | `-` | 25 | 1 | - | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 9:16 Portrait 5s i2v (25-step)** (`fl2va`) | `-` | 25 | 1 | 1800.694 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **512p 16:9 15s i2v (4-step)** (`fl2va`) | `-` | 4 | 1 | 1800.424 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_bf16_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 15s i2v (4-step, c=2)** (`fl2va`) | `-` | 4 | 2 | - | **-** | - | **0.00** | 0.00x | `0/4` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 15s i2v (4-step)** (`fl2va`) | `-` | 4 | 1 | 49.81 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 15s t2va (4-step)** (`t2va`) | `-` | 4 | 1 | 1800.438 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 15s i2v (12-step)** (`fl2va`) | `-` | 12 | 1 | - | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 15s i2v (25-step)** (`fl2va`) | `-` | 25 | 1 | - | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 15s i2v (49-step full)** (`fl2va`) | `-` | 49 | 1 | - | **-** | - | **0.00** | 0.00x | `0/1` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 5s i2v (4-step)** (`fl2va`) | `-` | 4 | 1 | 1800.711 | **-** | - | **0.00** | 0.00x | `0/3` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 5s t2va (25-step)** (`t2va`) | `-` | 25 | 1 | 1800.446 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 5s t2va (49-step full)** (`t2va`) | `-` | 49 | 1 | - | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 10s i2v (4-step)** (`fl2va`) | `-` | 4 | 1 | 1800.747 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 10s i2v (25-step)** (`fl2va`) | `-` | 25 | 1 | - | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 9:16 Portrait 5s i2v (25-step)** (`fl2va`) | `-` | 25 | 1 | 1800.143 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **512p 16:9 15s i2v (4-step)** (`fl2va`) | `-` | 4 | 1 | 1800.692 | **-** | - | **0.00** | 0.00x | `0/2` |
| `mh3_fp8_u4_sdpa` | `MiniMax-H3 (FL2VA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 15s i2v (4-step, c=2)** (`fl2va`) | `-` | 4 | 2 | - | **-** | - | **0.00** | 0.00x | `0/4` |
| `mh3_bf16_u4_sdpa_adaln` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA + AdaLN-online` | **768p 16:9 15s i2v (4-step)** (`fl2va`) | `1344x768` | 4 | 1 | 1523.544 | **325.317** | 80498.0 | **0.18** | 0.05x | `2/2` |
| `mh3_bf16_u4_sdpa_adaln` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA + AdaLN-online` | **768p 16:9 15s i2v (25-step)** (`fl2va`) | `1344x768` | 25 | 1 | - | **791.673** | 31526.3 | **0.08** | 0.02x | `2/2` |
| `mh3_bf16_u4_sdpa_adaln` | `MiniMax-H3 (FL2VA)` | `BF16` | 4 | `XLA Fused SDPA + AdaLN-online` | **768p 16:9 5s i2v (4-step)** (`fl2va`) | `1344x768` | 4 | 1 | 546.675 | **207.403** | 51292.8 | **0.29** | 0.02x | `2/2` |
| `fasth3_bf16_u4_sdpa` | `FastH3 (4-step VSA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 15s FastH3 (4-step VSA distill)** (`t2va`) | `-` | 4 | 1 | 1800.744 | **-** | - | **0.00** | 0.00x | `0/3` |
| `fasth3_bf16_u4_sdpa` | `FastH3 (4-step VSA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 10s FastH3 (4-step VSA distill)** (`t2va`) | `-` | 4 | 1 | 1800.16 | **-** | - | **0.00** | 0.00x | `0/3` |
| `fasth3_bf16_u4_sdpa` | `FastH3 (4-step VSA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 5s FastH3 (4-step VSA distill)** (`t2va`) | `-` | 4 | 1 | 1800.403 | **-** | - | **0.00** | 0.00x | `0/3` |
| `fasth3_bf16_u4_sdpa` | `FastH3 (4-step VSA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 9:16 Portrait 15s FastH3 (4-step)** (`t2va`) | `-` | 4 | 1 | 1800.569 | **-** | - | **0.00** | 0.00x | `0/2` |
| `fasth3_bf16_u4_sdpa` | `FastH3 (4-step VSA)` | `BF16` | 4 | `XLA Fused SDPA` | **512p 16:9 15s FastH3 (4-step)** (`t2va`) | `-` | 4 | 1 | 1800.209 | **-** | - | **0.00** | 0.00x | `0/3` |
| `fasth3_bf16_u4_sdpa` | `FastH3 (4-step VSA)` | `BF16` | 4 | `XLA Fused SDPA` | **768p 16:9 15s FastH3 (4-step, c=2)** (`t2va`) | `-` | 4 | 2 | - | **-** | - | **0.00** | 0.00x | `0/4` |
| `fasth3_fp8_u4_sdpa` | `FastH3 (4-step VSA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 15s FastH3 (4-step VSA distill)** (`t2va`) | `-` | 4 | 1 | 1800.17 | **-** | - | **0.00** | 0.00x | `0/3` |
| `fasth3_fp8_u4_sdpa` | `FastH3 (4-step VSA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 10s FastH3 (4-step VSA distill)** (`t2va`) | `-` | 4 | 1 | 1800.099 | **-** | - | **0.00** | 0.00x | `0/3` |
| `fasth3_fp8_u4_sdpa` | `FastH3 (4-step VSA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 5s FastH3 (4-step VSA distill)** (`t2va`) | `-` | 4 | 1 | 1800.421 | **-** | - | **0.00** | 0.00x | `0/3` |
| `fasth3_fp8_u4_sdpa` | `FastH3 (4-step VSA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 9:16 Portrait 15s FastH3 (4-step)** (`t2va`) | `-` | 4 | 1 | 1800.695 | **-** | - | **0.00** | 0.00x | `0/2` |
| `fasth3_fp8_u4_sdpa` | `FastH3 (4-step VSA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **512p 16:9 15s FastH3 (4-step)** (`t2va`) | `-` | 4 | 1 | 1800.125 | **-** | - | **0.00** | 0.00x | `0/3` |
| `fasth3_fp8_u4_sdpa` | `FastH3 (4-step VSA)` | `FP8 (online)` | 4 | `XLA Fused SDPA` | **768p 16:9 15s FastH3 (4-step, c=2)** (`t2va`) | `-` | 4 | 2 | - | **-** | - | **0.00** | 0.00x | `0/4` |

## 2. Full `v5p-16` Slice (`16x TPU v5p Chips` = `4×` Data-Parallel Replicas, `DP=4 × Ulysses=4`) Throughput & `30 RPM` Sizing

| Cluster Config | Workload / Input Distribution | Steps | Concurrency | Wall (s) | Med E2E (s) | P90 E2E (s) | Slice RPM (vids/min) | Video-s / Wall-s | Slices for 30 RPM | OK |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `DP=4 × Ulysses=4 (16 v5p chips, BF16, XLA SDPA)` | **512p 16:9 15s i2v (4-step)** | 4 | 4 | 120.879 | **-** | - | **0.00** | 0.00x | `-` | `0/8` |
| `DP=4 × Ulysses=4 (16 v5p chips, BF16, XLA SDPA)` | **512p 16:9 15s i2v (4-step)** | 4 | 8 | 120.773 | **-** | - | **0.00** | 0.00x | `-` | `0/12` |
| `DP=4 × Ulysses=4 (16 v5p chips, BF16, XLA SDPA)` | **512p 16:9 15s i2v (25-step)** | 25 | 4 | 120.862 | **-** | - | **0.00** | 0.00x | `-` | `0/8` |
| `DP=4 × Ulysses=4 (16 v5p chips, BF16, XLA SDPA)` | **512p 16:9 5s i2v (4-step)** | 4 | 4 | 121.539 | **-** | - | **0.00** | 0.00x | `-` | `0/12` |
| `DP=4 × Ulysses=4 (16 v5p chips, BF16, XLA SDPA)` | **512p 16:9 5s i2v (4-step)** | 4 | 8 | 120.863 | **-** | - | **0.00** | 0.00x | `-` | `0/16` |
| `DP=4 × Ulysses=4 (16 v5p chips, BF16, XLA SDPA)` | **Mixed Dist 50/30/20 (5s/10s/15s 512p i2v, 4-step)** | 4 | 4 | 120.862 | **-** | - | **0.00** | 0.00x | `-` | `0/10` |
| `DP=4 × Ulysses=4 (16 v5p chips, BF16, XLA SDPA)` | **Mixed Dist 50/30/20 (5s/10s/15s 512p i2v, 4-step)** | 4 | 8 | 120.867 | **-** | - | **0.00** | 0.00x | `-` | `0/16` |
