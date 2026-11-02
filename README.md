# TPU `v5p` Video + Audio Diffusion Serving

[![Benchmark Report](https://img.shields.io/badge/Benchmark-Report-blue)](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md)

Serving infrastructure, containers, Kubernetes Indexed Job manifests, and multi-configuration benchmark harnesses for **multimodal video + stereo audio diffusion models** on **Cloud TPU `v5p`** (`95 GiB` HBM2e per chip / `380 GiB` HBM2e per 4-chip host, `459` BF16 TFLOP/s per chip) using **`vllm-torchtpu` (`vLLM-Omni` v0.32.0)** and **`sglang-torchtpu` (`TpuOOTPlatform`)**:

- **Benchmark Report:** [`V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md`](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md)

---

## 1. Summary Results & 30-RPM Capacity Sizing

| Model & Workload | Engine & Packing | Warm Median Latency | Host RPM (`4` Chips) | Hosts (`Chips`) for `30 RPM` | Script & Line Anchor | K8s Manifest | Result Files |
|---|---|---:|---:|---|---|---|---|
| **`LTX-2.5` `544p` `5s 121f` (8-step High Density)** | **`vllm-omni` `2 × USP=2` / host** *(2-chip packing)* | **`6.931s`** *(85.6 RPM on 6 hosts)* | **`17.30`** *(+33.8% vs USP=4)* | **`2 hosts` (`8 chips`)** | [`omni.py#L208`](./scripts/run_v5p_ltx25_wan22_omni.py#L208-L345) · [`#L610`](./scripts/run_v5p_ltx25_wan22_omni.py#L610-L720) | [`omni-p2-dp6.yaml`](./k8s/v5p-32-spot-omni-serving-p2-dp6.yaml) | [`Report`](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md) · [`DP6 JSON`](./results/v5p32_spot_omni_serve_p2_dp6/summary.json) · [`Pod0`](./results/v5p32_spot_omni_serve_p2_dp6/pod_0_results.json) |
| **`LTX-2.5` `544p` `5s 121f` (8-step Low Latency)** | **`vllm-omni` `1 × USP=4` / host** | **`4.624s`** *(`1.08x` real-time)* | **`12.97`** | **`3 hosts` (`12 chips`)** | [`omni.py#L425`](./scripts/run_v5p_ltx25_wan22_omni.py#L425-L535) | [`omni-serving.yaml`](./k8s/v5p-32-spot-omni-serving.yaml) | [`Report`](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md) · [`Pod0 JSON`](./results/v5p32_spot_omni_serve_p23/pod_0_results.json) |
| **`LTX-2.5` `720p` `5s 121f` (8-step)** | **`vllm-omni` `2 × USP=2`** *(or `USP=4`)* | **`12.200s`** *(`7.383s` @ `USP=4`)* | **`9.84`** *(`8.13` @ `USP=4`)* | **`4 hosts` (`16 chips`)** | [`omni.py#L610`](./scripts/run_v5p_ltx25_wan22_omni.py#L610-L720) | [`omni-p2-dp6.yaml`](./k8s/v5p-32-spot-omni-serving-p2-dp6.yaml) | [`Report`](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md) · [`Pod0 JSON`](./results/v5p32_spot_omni_serve_p23/pod_0_results.json) |
| **`LTX-2.5` `544p` `10s 241f` (8-step)** | **`vllm-omni` `1 × USP=4` / host** | **`8.681s`** *(`1.15x` real-time)* | **`6.91`** | **`5 hosts` (`20 chips`)** | [`omni.py#L610`](./scripts/run_v5p_ltx25_wan22_omni.py#L610-L720) | [`omni-serving.yaml`](./k8s/v5p-32-spot-omni-serving.yaml) | [`Report`](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md) · [`Pod0 JSON`](./results/v5p32_spot_omni_serve_p23/pod_0_results.json) |
| **`Wan2.2-T2V` `480p` `5s 81f` (4-step fast)** | **`vllm-omni` `CFG=2 × USP=2`** *(or `USP=4`)* | **`27.108s`** *(`27.28s` @ `USP=4`)* | **`2.27`** | **`14 hosts` (`56 chips`)** | [`omni.py#L610`](./scripts/run_v5p_ltx25_wan22_omni.py#L610-L720) | [`omni-serving.yaml`](./k8s/v5p-32-spot-omni-serving.yaml) | [`Report`](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md) · [`Pod1 JSON`](./results/v5p32_spot_omni_serve_p23/pod_1_results.json) |
| **`Wan2.2-I2V` `480p` `5s 81f` (4-step fast)** | **`vllm-omni` `1 × USP=4` / host** | **`28.049s`** | **`2.18`** | **`14 hosts` (`56 chips`)** | [`omni.py#L561`](./scripts/run_v5p_ltx25_wan22_omni.py#L561-L609) | [`omni-serving.yaml`](./k8s/v5p-32-spot-omni-serving.yaml) | [`Report`](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md) · [`Pod1 JSON`](./results/v5p32_spot_omni_serve_p23/pod_1_results.json) |
| **`Wan2.2-T2V` `720p` `5s 81f` (4-step fast)** | **`vllm-omni` `CFG=2 × USP=2`** | **`44.223s`** *(`44.66s` @ `USP=4`)* | **`1.36`** | **`23 hosts` (`92 chips`)** | [`omni.py#L610`](./scripts/run_v5p_ltx25_wan22_omni.py#L610-L720) | [`omni-serving.yaml`](./k8s/v5p-32-spot-omni-serving.yaml) | [`Report`](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md) · [`Pod1 JSON`](./results/v5p32_spot_omni_serve_p23/pod_1_results.json) |
| **`Wan2.2-T2V` `480p` `5s 81f` (`20-step` / `40-step`)** | **`sglang-torchtpu` `Ulysses=4`** | **`118.538s`** *(`20s`)* / **`226.198s`** *(`40s`)* | **`0.51`** / **`0.27`** | **`59` / `112` hosts** | [`sglang.py#L214`](./scripts/run_v5p_ltx25_wan22_sglang.py#L214-L530) · [`#L928`](./scripts/run_v5p_ltx25_wan22_sglang.py#L928-L1045) | [`sglang-serving.yaml`](./k8s/v5p-32-spot-sglang-serving.yaml) | [`Report`](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md) · [`Pod0 JSON`](./results/v5p32_spot_sglang_serve_p23/pod_0_results.json) |
| **`MiniMax-H3 (FL2VA)` `768p` `5s` (4-step)** | **`sglang-torchtpu` `Ulysses=4`** | **`291.180s`** *(`58.2 GB` HBM)* | **`0.206`** | **`146 hosts` (`584 chips`)** | [`minimax.py#L1173`](./scripts/run_v5p_minimax_h3_serving.py#L1173-L1358) | [`minimax-p2.yaml`](./k8s/v5p-16-spot-minimax-h3-serving-p2.yaml) | [`P1 MD`](./results/v5p16_spot_minimax_h3_serve_p1/SUMMARY.md) · [`P2 JSON`](./results/v5p16_spot_minimax_h3_serve_p2/pod_0_results.json) |
| **`MiniMax-H3 (FL2VA)` `768p` `15s` (4-step HD)** | **`sglang-torchtpu` `Ulysses=4` (`p2` Tuned)** | **`385.956s`** *(from `458.69s` `p1`)* | **`0.155`** | **`194 hosts` (`776 chips`)** | [`minimax.py#L832`](./scripts/run_v5p_minimax_h3_serving.py#L832-L912) · [`#L1173`](./scripts/run_v5p_minimax_h3_serving.py#L1173-L1358) | [`minimax-p2.yaml`](./k8s/v5p-16-spot-minimax-h3-serving-p2.yaml) | [`P1 MD`](./results/v5p16_spot_minimax_h3_serve_p1/SUMMARY.md) · [`P2 Pod0`](./results/v5p16_spot_minimax_h3_serve_p2/pod_0_results.json) |

---

## 2. Optimization Summary

1. **`LTX-2.5` Sub-Host `2 × USP=2` Packing (`12.97` to `17.30 RPM/host`, `+33.8%`):**
   - Configured `auto_pad=True` in `SequenceParallelInput` and PyAV `1024`-sample AAC packet framing ([`apply_vllm_omni_tpu_patches#L208`](./scripts/run_v5p_ltx25_wan22_omni.py#L208-L345)), and packed two `2`-chip `USP=2` replicas per 4-chip `v5p` host ([`k8s/v5p-32-spot-omni-serving-p2-dp6.yaml`](./k8s/v5p-32-spot-omni-serving-p2-dp6.yaml)), reducing `30 RPM` fleet requirements from `12` chips (`3` hosts) to `8` chips (`2` hosts).
2. **`MiniMax-H3` (`33.14B` DiT + `Qwen3-VL`) Resident on 1 `v5p` Host (`458.69s` to `385.96s` for `15s 768p` Video+Audio):**
   - Applied 9 TPU runtime patches (Qwen3-VL contiguous slice splice, CPU `get_rope_index` `int32` lowering, selective encoder/VAE CPU offload `--dit-cpu-offload=false`, and precomputed/tuned AdaLN in [`scripts/run_v5p_minimax_h3_serving.py#L832`](./scripts/run_v5p_minimax_h3_serving.py#L832-L1018)), keeping the `66.3 GB` DiT resident in `v5p` HBM (`70.3–74.6 GB` peak) and reducing 15s HD generation latency from `458.69s` ([`p1`](./results/v5p16_spot_minimax_h3_serve_p1/SUMMARY.md)) to `385.96s` ([`p2`](./results/v5p16_spot_minimax_h3_serve_p2/pod_0_results.json)).

---

## 3. Repository Structure & Quickstart

- [`V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md`](./V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md) — Complete Phase 1 + Phase 2 benchmark report & `v7x` vs `v5p` comparison
- [`docker/Dockerfile.sglang_tpu`](./docker/Dockerfile.sglang_tpu), [`docker/prep_sglang_tpu.py`](./docker/prep_sglang_tpu.py), [`docker/Dockerfile.vllm_omni`](./docker/Dockerfile.vllm_omni), [`docker/prep_vllm_omni.py`](./docker/prep_vllm_omni.py)
- [`scripts/run_v5p_ltx25_wan22_omni.py`](./scripts/run_v5p_ltx25_wan22_omni.py) — `vllm-omni` single-host (`USP=4`), sub-host (`2 × USP=2`), `CFG=2 × USP=2` & multi-host `DP=6` harness
- [`scripts/run_v5p_ltx25_wan22_sglang.py`](./scripts/run_v5p_ltx25_wan22_sglang.py) — `sglang-torchtpu` `LTX-2.5` & `Wan 2.2` (`4/20/40`-step) harness
- [`scripts/run_v5p_minimax_h3_serving.py`](./scripts/run_v5p_minimax_h3_serving.py) — `MiniMax-H3` (`FL2VA` `i2v` & `t2va`) & `FastH3` `v5p-16` harness
- [`scripts/stage_all_video_models_gcs.py`](./scripts/stage_all_video_models_gcs.py) — Parallel HF Hub -> regional GCS checkpoint stager
- [`k8s/`](./k8s/) — [`v5p-32-spot-omni-serving-p2-dp6.yaml`](./k8s/v5p-32-spot-omni-serving-p2-dp6.yaml), [`v5p-32-spot-omni-serving.yaml`](./k8s/v5p-32-spot-omni-serving.yaml), [`v5p-32-spot-sglang-serving.yaml`](./k8s/v5p-32-spot-sglang-serving.yaml), [`v5p-16-spot-minimax-h3-serving-p2.yaml`](./k8s/v5p-16-spot-minimax-h3-serving-p2.yaml), [`v5p-16-spot-minimax-h3-serving.yaml`](./k8s/v5p-16-spot-minimax-h3-serving.yaml)

```bash
# 1. Upload serving benchmark harnesses to regional GCS
gcloud storage cp scripts/run_v5p_*.py gs://YOUR_GCS_BUCKET/v5p-video-serving-scripts/

# 2. Run LTX-2.5 2xUSP=2 & Multi-Host DP=6 (85.6 RPM) + Wan 2.2 on tpu-v5p-32-pool
kubectl apply -f k8s/v5p-32-spot-omni-serving-p2-dp6.yaml
kubectl apply -f k8s/v5p-32-spot-sglang-serving.yaml

# 3. Run Tuned Resident MiniMax-H3 (385.96s 15s 768p HD) on tpu-v5p-16-pool
kubectl apply -f k8s/v5p-16-spot-minimax-h3-serving-p2.yaml
```
