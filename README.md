# Cloud TPU `v5p` (`2x2x1` Single-Host, `2x2x4` `v5p-16` & `2x4x4` `v5p-32` Slice) Video+Audio Diffusion Serving (`MiniMax-H3`, `LTX-2.5`, `Wan 2.2 A14B`)

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

## 1. Key TPU `v5p` Serving Architecture & Optimizations

| Feature | Implementation | Impact |
|---|---|---|
| **Single-Host `2x2x1` Isolation on Multi-Host `2x2x4` / `2x4x4` Slices** | `LD_PRELOAD=/workspace/metadata_hook.so` + local `v5p`-aware `mock_metadata.py` (`TPU_ACCELERATOR_TYPE=v5p-8`, `CHIPS_PER_HOST_BOUNDS=2,2,1`) | Runs **4 independent `4`-chip (`380 GiB` HBM2e) servers** on `v5p-16` (`DP=4 × Ulysses=4`) and **8 independent `4`-chip servers** on `v5p-32` (`DP=8 × Ulysses=4`) with zero cross-host ICI barriers |
| **5-Point `MiniMax-H3` `sglang-torchtpu` TPU Compatibility Patchset** | TPU `ProcessGroupTPU` AdaLN device placement, non-CUDA VDN attention branch execution, CPU `get_rope_index` + `int32` token casting, static contiguous slice splice in `Qwen3-VL` deepstack/visual tokens, and `tpu` denoising stage enablement | Unlocks full `MiniMax-H3` (`FL2VA`: `t2va` & `fl2va` `i2v`) and `FastH3` 4-step VSA joint video+audio serving on `torch_tpu` |
| **`95 GiB`/chip HBM Budget Management (`v5p` Host = `380 GiB` HBM)** | `--encoder-parallel fold` (`4`-way sharded `Qwen3-VL` `32B`) + `--text-encoder-cpu-offload` (or Online `FP8` `--quantization fp8`) | Comfortably fits `66.3 GB` `MiniMax-H3` DiT + `Qwen3-VL` + `3D Video VAE` + `Audio VAE` inside `380 GiB` single-host HBM |
| **TPU Seeded Generator Deadlock Prevention** | Passes `"generator_device": "cpu"` in `/v1/videos` requests | Avoids GIL deadlock in blocking host-to-device RNG seed copy on `torch_tpu` |
| **GCSFuse + `/dev/shm` Single-Active-Model Staging** | Streams active checkpoint shards into `/dev/shm` and purges between model families | Prevents `448 GiB` `ct5p-hightpu-4t` host RAM exhaustion while achieving multi-GB/s model load |

---

## 2. Repository Structure

```text
├── README.md
├── docker/
│   ├── Dockerfile.sglang_tpu                      # TorchTPU 0.2.0 + sglang-torchtpu + varlen patch + ffmpeg
│   ├── prep_sglang_tpu.py                         # ABI constraint lock + TOML-verified sglang-torchtpu setup
│   ├── Dockerfile.vllm_omni                       # TorchTPU 0.2.0 + vllm-torchtpu v0.32.0 + vLLM-Omni (CPU/TPU target)
│   └── prep_vllm_omni.py                          # ABI constraint lock for vLLM-Omni on TPU
├── scripts/
│   ├── stage_all_video_models_gcs.py              # Parallel HF Hub -> regional GCS stager for MiniMax-H3, FastH3, LTX-2.5, Wan 2.2
│   ├── run_v5p_minimax_h3_serving.py              # v5p-16: MiniMax-H3 (FL2VA i2v + t2va) & FastH3 4-step VSA benchmark suite
│   └── run_v5p_ltx25_wan22_serving.py             # v5p-32: LTX-2.5 (incl. 1080p 20s Kling-V4 proxy) & Wan 2.2 A14B across sglang & vLLM-Omni
├── k8s/
│   ├── v5p-16-spot-minimax-h3-serving.yaml        # Indexed Job (4x4=16 v5p chips) on ank-tpu-v5p-16-spot
│   ├── v5p-32-spot-ltx25-wan22-sglang.yaml        # Indexed Job on ank-tpu-v5p-32-spot for sglang-torchtpu + DP=8 cluster
│   └── v5p-32-spot-ltx25-wan22-vllm-omni.yaml     # Indexed Job on ank-tpu-v5p-32-spot for vllm-omni comparative serving
└── results/
    └── V5P_VIDEO_SERVING_BENCHMARK_RESULTS.md     # Comprehensive benchmark report & 30-RPM slice sizing
```

---

## 3. Quickstart

```bash
# 1. Upload serving benchmark harnesses to regional GCS
gcloud storage cp scripts/run_v5p_*.py gs://ank-usc1-bucket-314837540096/v5p-video-serving-scripts/

# 2. Run MiniMax-H3 (FL2VA i2v + t2va) & FastH3 on ank-tpu-v5p-16-spot (4 hosts = 16 v5p chips)
kubectl --context=gke_diesel-patrol-382622_us-central1_ank-tpu-v5p-usc1 apply -f k8s/v5p-16-spot-minimax-h3-serving.yaml

# 3. Run LTX-2.5 & Wan 2.2 A14B comparative serving on ank-tpu-v5p-32-spot (8 hosts = 32 v5p chips)
kubectl --context=gke_diesel-patrol-382622_us-central1_ank-tpu-v5p-usc1 apply -f k8s/v5p-32-spot-ltx25-wan22-sglang.yaml
kubectl --context=gke_diesel-patrol-382622_us-central1_ank-tpu-v5p-usc1 apply -f k8s/v5p-32-spot-ltx25-wan22-vllm-omni.yaml
```
