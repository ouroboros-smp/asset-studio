"""Qwen-Image-2512 quant/step benchmark through a local ComfyUI (used for the fp8 / NVFP4 / Lightning comparison in
docs/RESULTS.md). Not part of the app: the app runs diffusers in its own container. Usage: python comfy_bench.py <config>... | all
Env: COMFY_URL (default http://127.0.0.1:8188), COMFY_OUTPUT (ComfyUI output dir), BENCH_DEST (where results are copied)."""
import json, sys, time, urllib.request, uuid, os, shutil
from pathlib import Path

API = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
OUT = Path(os.environ.get("COMFY_OUTPUT", "comfyui/output"))  # set COMFY_OUTPUT to your ComfyUI output folder
DEST = Path(os.environ.get("BENCH_DEST", "bench-out/qwen2512"))
DEST.mkdir(parents=True, exist_ok=True)
PROMPT = open(os.path.join(os.path.dirname(__file__), "prompt_toaster.txt"), encoding="utf-8").read().strip()
NEG = open(os.path.join(os.path.dirname(__file__), "negative.txt"), encoding="utf-8").read().strip()
TE = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
SEEDS = [1523438529, 1523438530]

CONFIGS = {
    "bf16_50":       dict(unet="qwen_image_2512_bf16.safetensors", lora=None, steps=50, cfg=4.0),
    "bf16_light8":   dict(unet="qwen_image_2512_bf16.safetensors", lora="Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors", steps=8, cfg=1.0),
    "fp8_50":        dict(unet="qwen_image_2512_fp8_e4m3fn.safetensors", lora=None, steps=50, cfg=4.0),
    "nvfp4_50":      dict(unet="qwen-image-2512-nvfp4-v2.safetensors", lora=None, steps=50, cfg=4.0),
    "fp8_light8":    dict(unet="qwen_image_2512_fp8_e4m3fn.safetensors", lora="Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors", steps=8, cfg=1.0),
    "nvfp4_light8":  dict(unet="qwen-image-2512-nvfp4-v2.safetensors", lora="Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors", steps=8, cfg=1.0),
    "fp8_light4":    dict(unet="qwen_image_2512_fp8_e4m3fn.safetensors", lora="Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors", steps=4, cfg=1.0),
    "nvfp4_light4":  dict(unet="qwen-image-2512-nvfp4-v2.safetensors", lora="Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors", steps=4, cfg=1.0),
}

def workflow(c, seed, batch=1, w=1328, h=1328, tag="x"):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": c["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 3.1, "model": ["1", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["2", 0]}},
        "8": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": batch}},
        "9": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": c["steps"], "cfg": c["cfg"], "sampler_name": "euler",
                                                    "scheduler": "simple", "denoise": 1.0, "model": ["5", 0],
                                                    "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["8", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"bench/{tag}", "images": ["10", 0]}},
    }
    if c["lora"]:
        g["4"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": c["lora"], "strength_model": 1.0, "model": ["1", 0]}}
        g["5"]["inputs"]["model"] = ["4", 0]
    return g

def post(path, data):
    req = urllib.request.Request(API + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def get(path):
    return json.loads(urllib.request.urlopen(API + path).read())

def run(name, seed, batch=1, w=1328, h=1328):
    tag = f"{name}_s{seed}_b{batch}_{w}"
    t0 = time.time()
    pid = post("/prompt", {"prompt": workflow(CONFIGS[name], seed, batch, w, h, tag), "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        h_ = get(f"/history/{pid}")
        if pid in h_:
            break
        time.sleep(0.5)
    dt = time.time() - t0
    st = h_[pid]["status"]
    ok = st.get("status_str") == "success"
    files = []
    if ok:
        for o in h_[pid]["outputs"].values():
            for im in o.get("images", []):
                src = OUT / im["subfolder"] / im["filename"]
                dst = DEST / im["filename"]
                shutil.copy2(src, dst)
                files.append(dst.name)
    else:
        msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
        print("ERROR", json.dumps(msgs)[:1500])
    rec = {"config": name, "seed": seed, "batch": batch, "res": w, "wall_s": round(dt, 1), "ok": ok, "files": files, **CONFIGS[name]}
    print(json.dumps(rec), flush=True)
    with open(DEST / "results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    if names == ["all"]:
        names = list(CONFIGS)
    for n in names:
        for s in SEEDS:  # first run pays model load; second is steady state
            run(n, s)
