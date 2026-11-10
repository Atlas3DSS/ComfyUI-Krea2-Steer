#!/usr/bin/env python3
"""Fixed-seed comparison sheets for steering directions, through any running ComfyUI that has this pack loaded.

    lab.py PLAN.json OUT_DIR [--server http://127.0.0.1:8188] [--unet NAME] [--clip NAME] [--vae NAME]

PLAN.json:
    {"size": [832, 1216], "steps": 8, "seed": 101, "fit_size": [768, 1024],
     "directions": {"name": {"toward": "...", "away": "...", "text": true, "image": true}},
     "tests": {"name": "prompt"},
     "arms": [{"name": "base"},
              {"name": "x", "direction": "name", "text": -4, "image": -2, "blocks": [0, 27], "sigmas": [1.0, 0.0]},
              {"name": "typed", "prefix": "words put before the prompt"}]}

Every arm renders every test with the same seed, so rows differ only by the steering. Finished pictures are skipped.
Writes OUT_DIR/out/<arm>/<test>.png, sheet-whole.jpg, sheet-crop.jpg (a full-size centre crop) and detail.json
(pixel-scale contrast of that crop: pores and grain raise it, smoothing lowers it). Prompts are never printed.
"""
import argparse, json, os, time, urllib.error, urllib.parse, urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("plan")
parser.add_argument("out")
parser.add_argument("--server", default="http://127.0.0.1:8188")
parser.add_argument("--unet", default="krea2_turbo_bf16.safetensors")
parser.add_argument("--clip", default="qwen3vl_4b_bf16.safetensors")
parser.add_argument("--vae", default="qwen_image_vae.safetensors")
parser.add_argument("--sheet-only", action="store_true")
args = parser.parse_args()
PLAN = json.load(open(args.plan))
SIZE, STEPS, SEED = PLAN.get("size", [832, 1216]), PLAN.get("steps", 8), PLAN.get("seed", 101)
FIT_SIZE = PLAN.get("fit_size", [768, 1024])


def call(path, body=None, raw=False):
    request = urllib.request.Request(args.server + path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
    try:
        reply = urllib.request.urlopen(request, timeout=600).read()
    except urllib.error.HTTPError as error:  # ComfyUI explains a refused graph in the body
        detail = json.loads(error.read() or b"{}")
        errors = [e.get("details") or e.get("message") for node in detail.get("node_errors", {}).values() for e in node.get("errors", [])]
        raise RuntimeError(f"{error.code}: " + "; ".join(str(e) for e in errors or [detail.get("error", {}).get("message", "refused")])[:400]) from None
    return reply if raw else json.loads(reply)


def DEFAULT_SCENES(cache=[]):
    if not cache:
        cache.append(call("/object_info/Krea2SteerFitText")["Krea2SteerFitText"]["input"]["required"]["scenes"][1]["default"])
    return cache[0]


def log(text):
    print(time.strftime("%H:%M:%S"), text, flush=True)


def graph(arm, prompt):
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": args.unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": args.clip, "type": "krea2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": (arm.get("prefix", "") + " " + prompt).strip()}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": SIZE[0], "height": SIZE[1], "batch_size": 1}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    }
    model = ["unet", 0]
    name = arm.get("direction")
    if name:
        spec = PLAN["directions"][name]
        saved = f"{PLAN.get('prefix', 'lab')}/{name}.safetensors"
        if saved in call("/object_info/Krea2SteerLoad")["Krea2SteerLoad"]["input"]["required"]["name"][0]:
            # Fitted earlier in this plan (or an earlier run): load the file instead of fitting again.
            g["load"] = {"class_type": "Krea2SteerLoad", "inputs": {"name": saved}}
            steer = ["load", 0]
        else:
            steer = None
            # The API does not fill in defaults: take the node's own scene list unless the plan gives one.
            common = {"model": ["unet", 0], "clip": ["clip", 0], "toward": spec["toward"], "away": spec.get("away", ""), "save_as": "",
                      "scenes": "\n".join(spec["scenes"]) if "scenes" in spec else DEFAULT_SCENES()}
            if spec.get("text", True):
                g["fit_text"] = {"class_type": "Krea2SteerFitText", "inputs": dict(common)}
                steer = ["fit_text", 0]
            if spec.get("image", True):
                inputs = dict(common, width=FIT_SIZE[0], height=FIT_SIZE[1], steps=STEPS, sampler_name="euler", scheduler="simple", seed=1)
                if steer:
                    inputs["steer"] = steer
                g["fit_image"] = {"class_type": "Krea2SteerFitImage", "inputs": inputs}
                steer = ["fit_image", 0]
            g[steer[0]]["inputs"]["save_as"] = saved  # the last fit node holds both parts
        blocks, sigmas = arm.get("blocks", [0, 27]), arm.get("sigmas", [1.0, 0.0])
        g["apply"] = {"class_type": "Krea2SteerApply", "inputs": {
            "model": model, "steer": steer, "text_strength": arm.get("text", 0), "image_strength": arm.get("image", 0),
            "first_block": blocks[0], "last_block": blocks[1], "start_sigma": sigmas[0], "end_sigma": sigmas[1]}}
        model = ["apply", 0]
    g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["text", 0]}}
    g["sigmas"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": STEPS, "denoise": 1.0}}
    g["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "steer-lab/img"}}
    return g


def render(arm, prompt):
    answer = call("/prompt", {"prompt": graph(arm, prompt)})
    if "prompt_id" not in answer:
        raise RuntimeError(json.dumps(answer.get("node_errors") or answer.get("error"))[:600])
    deadline = time.time() + 3600
    while time.time() < deadline:
        history = call("/history/" + answer["prompt_id"])
        if answer["prompt_id"] in history:
            record = history[answer["prompt_id"]]
            if record.get("status", {}).get("status_str") == "error":
                messages = [m[1] for m in record["status"].get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(json.dumps({k: messages[0].get(k) for k in ("node_type", "exception_message")} if messages else {})[:600])
            image = record["outputs"]["save"]["images"][0]
            return call("/view?" + urllib.parse.urlencode({k: image[k] for k in ("filename", "subfolder", "type")}), raw=True)
        time.sleep(1.0)
    raise TimeoutError("timed out")


def run():
    total, done = len(PLAN["arms"]) * len(PLAN["tests"]), 0
    for arm in PLAN["arms"]:
        folder = os.path.join(args.out, "out", arm["name"])
        os.makedirs(folder, exist_ok=True)
        for test, prompt in PLAN["tests"].items():
            done += 1
            path = os.path.join(folder, test + ".png")
            if os.path.exists(path):
                continue
            started = time.time()
            try:
                data = render(arm, prompt)
            except Exception as error:  # noqa: BLE001
                log(f"{arm['name']}/{test}: {type(error).__name__}: {str(error)[:500]}")
                continue
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            log(f"{done}/{total} {arm['name']}/{test} {time.time() - started:.0f}s")


def sheet():
    import numpy
    from PIL import Image, ImageDraw
    rows, tests = [a["name"] for a in PLAN["arms"]], list(PLAN["tests"])
    label, crop = 230, PLAN.get("crop", 384)
    cell = (208, round(208 * SIZE[1] / SIZE[0]))
    whole = Image.new("RGB", (label + cell[0] * len(tests), cell[1] * len(rows)), "white")
    close = Image.new("RGB", (label + crop * len(tests), crop * len(rows)), "white")
    detail = {}
    for r, arm in enumerate(rows):
        ImageDraw.Draw(whole).text((6, r * cell[1] + 8), arm, fill="black")
        ImageDraw.Draw(close).text((6, r * crop + 8), arm, fill="black")
        for c, test in enumerate(tests):
            path = os.path.join(args.out, "out", arm, test + ".png")
            if not os.path.exists(path):
                continue
            picture = Image.open(path).convert("RGB")
            whole.paste(picture.resize(cell), (label + c * cell[0], r * cell[1]))
            w, h = picture.size
            left, top = (w - crop) // 2, int(h * PLAN.get("crop_height", 0.36)) - crop // 2
            piece = picture.crop((left, top, left + crop, top + crop))
            close.paste(piece, (label + c * crop, r * crop))
            grey = numpy.asarray(piece.convert("L"), dtype=numpy.float32)
            lap = 4 * grey[1:-1, 1:-1] - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:]
            detail.setdefault(arm, {})[test] = round(float(lap.std()), 2)
    whole.save(os.path.join(args.out, "sheet-whole.jpg"), quality=88)
    close.save(os.path.join(args.out, "sheet-crop.jpg"), quality=92)
    json.dump(detail, open(os.path.join(args.out, "detail.json"), "w"), indent=1)
    for arm, values in detail.items():
        log(f"{arm:28s} " + " ".join(f"{k} {v:6.2f}" for k, v in values.items()))


os.makedirs(args.out, exist_ok=True)
if not args.sheet_only:
    run()
sheet()
log("done")
