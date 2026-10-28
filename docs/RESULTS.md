# Measured results (RTX 5090, 2026-09-12)

Host: Windows 11 Pro, single RTX 5090 (32 GB, driver 616.92), 96 GB RAM of which the Docker Desktop WSL2 VM gets 46 GiB.
All numbers below are from real jobs on this machine (`samples/*/manifest.json`, `samples/acceptance_report.json`).
Nothing here is an upstream H100 figure.

## Acceptance jobs

| Job | Preset | Reference (Qwen-Image-2512) | Pixal3D | Blender | Optimized result |
|---|---|---|---|---|---|
| crate `20260912-224921-e6af094a` | balanced (1024) | 1 candidate, 285 s | 243 s total: load 52 s, matting/crop 15 s, MoGe camera 5 s, **generate 112 s**, **export 56 s** (1M-tri / 4096² master) | 78 s | 7,998 tris via voxel proxy (1.4 % error), LODs 3,997/1,996, hull 200 |
| pump `20260912-231658-6581cb96` | **quality (1536)** | 2 candidates, 297 s + 288 s | 222 s total: load 54 s, **generate 85 s**, **export 77 s**; 20.5 M raw faces at grid 1536 | 80 s | 19,803 tris direct (2.3 % error), LODs 9,813/5,227, hull 200 |
| mug `20260912-231658-f8dcb166` | balanced (1024) | 1 candidate, 347 s stage | 250 s | 74 s | 6,000 tris direct (1.4 % error), LODs 3,088/1,500, handle opening preserved |

Wall clock for a complete balanced job (text → packaged asset) is about 11 minutes; quality with two candidates about 20 minutes.
Most of it is Qwen-Image at the model card's full-quality 50 steps (5.7–6 s/step at 1328² with 24 of 60 transformer blocks
CPU-offloaded, PCIe-bound). Pixal3D model loading (~52 s) is paid once per job because each stage is a fresh subprocess.

## Memory

| Stage | torch max reserved | nvidia-smi peak (whole GPU) | Process peak RSS |
|---|---|---|---|
| Qwen-Image-2512 BF16, 36/60 blocks GPU-resident (first runs) | 31.3–31.5 GB | 31.9–32.0 GB | 21.5–22.3 GB |
| Qwen-Image-2512 BF16, **30/60 blocks resident + VAE tiling (current default)** | ≈27 GB expected (4 blocks ≈ 2.7 GB less) | – | – |
| Pixal3D 1024 cascade, low_vram | generate 15.3 GB, export 12.4 GB | 19.2 GB | 27.3 GB |
| Pixal3D **1536** cascade, low_vram | generate **24.9 GB**, export 14.8 GB | 27.4 GB | 27.6 GB |
| Blender (CPU) | – | – | ~2–4 GB |

The first Qwen runs peaked at the 32 GB ceiling without failing; the default was lowered to 30 resident blocks and VAE
tiling enabled to keep headroom. Pixal3D in low-VRAM mode keeps its ~29 GB of weights in host RAM (RSS 27 GB) and moves one
model at a time to the GPU; standard mode would not fit 1536 on 32 GB alongside the DINOv3/NAF conditioning models.

## Effective quality settings

* Reference: 1328×1328, 50 steps, `true_cfg_scale` 4.0, negative prompt, BF16, no LoRA, seed = request seed (+i per candidate).
* Pixal3D: upstream sampler defaults (12 steps per stage, guidance 7.5/7.5/1.0, guidance intervals from `pipeline.json`),
  `pipeline_type = 1024_cascade | 1536_cascade`, `max_num_tokens` 49152, flash-attn backend, BiRefNet matting,
  MoGe-2 camera. Export adapter: `to_glb(decimation_target=1_000_000, texture_size=4096, remesh=True, remesh_band=1,
  remesh_project=0)` → PNG textures (`extension_webp=False`).
* No OOM fallback was triggered in any acceptance job (`fallbacks_applied: []`).

## Fast reference models (2026-09-13, same prompt/seed sweep)

Steady-state seconds per image and `torch.cuda.max_memory_reserved` inside the image worker (first image of a job also
pays the model load: Klein 4B ≈ 6 s, Klein 9B ≈ 30 s incl. the text-encoder → transformer swap, Z-Image ≈ 25 s after the
one-time copy of the 12 GB file into the Linux cache). Settings follow the vendors' reference configs; only the
resolution was swept. All outputs stayed coherent (single centred object, no artifacts) at every setting.

| Model | Setting | s / image | peak VRAM |
|---|---|---|---|
| FLUX.2 Klein 9B (fp8→bf16) | 1024², 4 steps, guidance 1.0 | 2.9 (22.8 incl. load) | 20.3 GB |
| | 1280², 4 steps | 3.0 | 22.1 GB |
| | **1536², 4 steps (default)** | 4.4 | 23.9 GB |
| | 2048², 4 steps | 9.1 | 29.1 GB |
| | 1536², 6 steps | 7.0 | – (no visible gain; distillation fixes 4 steps) |
| FLUX.2 Klein 4B (bf16) | 1024², 4 steps | 2.1 | 18.1 GB |
| | **1536², 4 steps (default)** | 2.4 | 21.8 GB |
| Z-Image Turbo (bf16) | 1024², 9 steps, guidance 0 (official) | 6.4 | 22.5 GB |
| | 1024², 8 steps, guidance 1.0 (old default) | 7.0 | 22.5 GB (CFG silently on: 2 passes/step, softer output) |
| | **1536², 9 steps, guidance 0 (default)** | 8.7 | 26.0 GB |

Why 1536² and not 2048²: the 1536² crops show noticeably finer paint wear and edge detail than 1024² for ~1.5 s extra,
while 2048² peaks at 29 GB and would collide with a concurrent Pixal3D run under the lenient GPU-sharing policy.

## Qwen-Image-2512: fp8 resident vs the old bf16 offload (2026-09-13)

The 20B transformer is 40 GB in bf16, so the original path kept 30 of 60 blocks on the GPU and streamed the rest over PCIe
twice per step (CFG): ~5.8 s/step, ~290 s per 50-step image. The app now stores the transformer as float8_e4m3fn with bf16
compute (diffusers layerwise casting), built once from the HF bf16 weights with the Lightning LoRA fused in bf16 first
(`services/image_worker/generate.py: _build_fp8_cache`, 53 s, cached as a 20.5 GB file in the models volume).

| Path | Steps | s / image (steady) | peak VRAM | note |
|---|---|---|---|---|
| bf16, 30/60 blocks offloaded (old default) | 50, CFG 4 | ~290 | 27-31 GB | PCIe-bound |
| **fp8 resident + Lightning 8 (new default)** | 8, no CFG | **8.0** (11 s first image) | 20.8 GB | transformer load 5 s from cache |
| fp8 resident + Lightning 4 | 4, no CFG | ~4 | 20.8 GB | |
| fp8 resident, undistilled | 50, CFG 4 | ~80 | ~21 GB | option only |

### Quantization / step comparison (ComfyUI test bench, same prompt and seeds, 1328²)

Run with `tools/bench/comfy_bench.py` against a local ComfyUI 0.35 to compare candidate weights before touching the app.
Warm seconds per image on the RTX 5090:

| Weights | 50 steps CFG 4 | Lightning 8 | Lightning 4 | verdict |
|---|---|---|---|---|
| Comfy-Org fp8 e4m3fn (plain cast) | 81 | 7.1 | 4.0 | clean, matches bf16 look |
| Hippotes NVFP4 v2 (community 4-bit) | 43 | 4.0 | 2.5 | grainy background, blotchy surfaces on one object, over-greebled with Lightning; rejected |
| lightx2v "fp8 scaled" file | – | – | – | loads as noise in ComfyUI (scales ignored); not usable there |

The 50-step CFG-4 output was rated *below* the Lightning output by eye on two objects: waxier, lower-contrast surfaces with
less material detail (`docs/bench_grinder_crops.jpg`). Lightning's distillation bakes in higher micro-contrast, which is a
plus for a Pixal3D reference image.

### Blind test (drone landing port, 7 models × 3 seeds, user picks, `tools/bench/blind_test.py`)

Scores: fav +2, ok +1, bad −2. Reveal sheet: `docs/bench_droneport_reveal.jpg`.

| Model | picks | score |
|---|---|---|
| Qwen-2512 fp8 + Lightning 8 | fav, ok, ok | +4 |
| Z-Image Turbo (app, 1536²) | ok, ok, ok | +3 |
| FLUX.2 Klein 4B (app, 1536²) | ok, ok, bad | 0 |
| Qwen-2512 NVFP4 + Lightning 8 | ok, bad, bad | −3 |
| FLUX.2 Klein 9B (app, 1536²) | ok, bad, bad | −3 |
| Qwen-2512 fp8 + Lightning 4 | ok, bad, bad | −3 |
| Qwen-2512 NVFP4 + Lightning 4 | bad, –, bad | −4 |

Lightning 4 vs 8 share seeds, so the compositions match and the difference is detail: 8 steps wins clearly for 3 s more.

## Reduction quality (what changed during acceptance)

1. Blender's collapse decimation on the ~1 M-triangle, ~200-shell Pixal3D masters produced spiky, holed low-polys and the
   pure-Python cleanup was O(minutes). Replaced.
2. `fast-simplification` hit the counts but thin planks still collapsed into holes below ~1 %.
3. **meshoptimizer** (`simplifyWithAttributes`, normals weight 0.7, `Prune`) with an error ladder 1 % → 3 % → 5 % of the
   mesh extent: pump 20k at 2.3 % error, mug 6k, both visually matching the master in the preview renders.
4. When the budget cannot be met within 5 % (crate at 8k: only reachable at 12 % error, planks open up), the copy is
   voxel-remeshed into one shell (voxel ≈ 0.45 % of the diagonal) and reduced from there; this is recorded in the manifest
   as `voxel_proxy_decimate_rebake` with a warning. Colour/metallic/roughness/normal are always baked from the untouched
   master; a hit-mask bake fills ray misses.

## Robustness tests (real, not mocked)

* **Cancel:** cancelling mid-Blender stopped the stage subprocess within seconds; status `cancelled`.
* **Restart recovery:** `docker compose restart worker` during the pump's Pixal3D stage → job requeued, reference stage reused
  (`reference stage already complete; reusing`), Pixal3D re-run, job completed. The orphaned stage process of the previous
  worker is now cancelled on startup (it initially just ran to completion while the new attempt waited).
* **Retry / reprocess:** `POST /v1/jobs/{id}/retry` resumes failed jobs from completed stages;
  `{"from_stage":"blender","optimize":{...}}` re-optimises a finished job (used to re-run all three assets with new budgets
  and the meshoptimizer path without regenerating images or 3D).
* **Two lanes, no VRAM thresholds (2026-09-13):** the worker runs an image lane and an asset lane, so an 8-second
  image session never waits behind a 4-minute 3D job. GPU stages start immediately; on CUDA OOM the stage waits for the
  other lane's GPU stage to finish and retries with the same settings (then one paused retry, then the quality ladder).
  Other GPU users (ComfyUI, host Blender viewports) are never killed.
* **Restart re-attach:** each GPU/Blender stage records its runner run id in `stages/<stage>/run.json`; a restarted worker
  re-attaches to a stage still running in the runner container instead of cancelling it (a worker rebuild used to cost a
  full Pixal3D stage).
* **Offline:** every worker runs with `HF_HUB_OFFLINE=1`; the Pixal3D and Qwen loaders were verified with
  `docker run --network none`.
* **Mocked error paths:** `tests/test_api_mock.py` + `tests/test_portal_mock.py` (20 tests): invalid inputs, auth, cancel, artifact path confinement,
  orphan requeue, stage resume + OOM fallback policy, retry, prompt rules, GLB validation incl. WebP rejection.
* **External checks:** Khronos glTF validator (`tools/validate_gltf.js`): 0 errors on all 15 produced GLBs (one info-level
  warning about runtime tangent generation on normal-mapped meshes); headless Godot 4.7.2 import (`tools/godot_import_test.py`)
  succeeded for every GLB. Unity is not installed on this machine; Godot is the engine importer
  that was actually exercised.

## Known limitations

* The reference-candidate scorer is technical only (framing, margins, clipping, background plainness); an external local
  vision evaluator can be plugged in (`STUDIO_VISION_EVALUATOR_URL`) but none is bundled. The first scorer (global colour
  threshold) misread the crate reference's soft vignette as clipping; the edge-based scorer that replaced it rates the same
  image 1.0 and scored both pump candidates 1.0.
* Matting: the three acceptance jobs used `ZhengPeng7/BiRefNet` (RMBG-2.0 access was granted later; it is now the default,
  `STUDIO_REMBG_MODEL` switches).
* Multi-view: the calibrated-input path is implemented and validated statically (matrices, FOV, front-view convention);
  the `*_mv` checkpoints were not downloaded (+17 GB) and no multi-view job was run. No automatic novel-view generation.
* No quantised Qwen mode yet (BF16 with block offload only); a faster mode would need a quality comparison first.
* Output is an optimized static asset: automatic decimation topology, no rig, not animation-ready.
* Text/lettering: models garble it; the prompt builder forbids text on the object.
