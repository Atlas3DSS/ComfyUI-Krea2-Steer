"""Steering vectors for Krea 2: the maths and the hooks, free of any node or UI code.

Krea 2 in one line:  prompt -> Qwen3-VL (12 hidden layers) -> txtfusion -> txtmlp -> text tokens (6144 wide)
                     [text tokens + image tokens] -> 28 shared transformer blocks -> picture

Two places take a constant vector, and a constant vector is all a steering direction is:

  text   the bias of ``txtmlp.3``, the last layer every text token passes before the blocks read it.
         One vector of 6144 numbers. It is a weight patch, so it costs nothing at run time.
  image  the image tokens leaving each block (the residual stream). One vector per block and per noise level,
         added while sampling. This is the part that can act on late steps only (texture) and leave early steps
         (layout) alone.

A direction is always a mean difference between two conditions ("toward" minus "away"): two wordings of the same
scenes, or two folders of pictures. Nothing is trained.
"""
from __future__ import annotations

import json

import torch

TEXT_BIAS_KEY = "diffusion_model.txtmlp.3.bias"
LORA_BIAS_KEY = "diffusion_model.txtmlp.3.diff_b"
WRAPPER_KEY = "krea2_steer"
FORMAT = "krea2-steer-1"

DEFAULT_SCENES = """A portrait of a woman in her thirties with short dark hair
An elderly man reading a newspaper on a park bench
A golden retriever running across a meadow
A tabby cat asleep on a windowsill
A bowl of ramen with a soft egg and spring onions
A mountain lake at sunrise with pine trees
A living room with a sofa, a rug and a floor lamp
A red bicycle leaning against a brick wall
A man in his forties in a grey suit walking through a station
A woman of fifty in a yellow raincoat holding an umbrella
A glass vase of tulips on a table
A busy market street with fruit stalls
A chef of sixty plating a dish in a restaurant kitchen
A stack of old books beside a cup of tea
A close-up of a man of thirty with a beard, looking at the camera
A close-up of a woman of sixty-five smiling, grey hair tied back
A woman of twenty-eight with freckles sitting by a window
A man of fifty in a wool jumper leaning on a fence
Two women in their forties laughing at a cafe table
A swimmer of thirty-five resting at the edge of a pool
A carpenter of forty-five sanding a table in a workshop
A bride of thirty adjusting her veil in a mirror
A nurse of fifty-five in scrubs standing in a corridor
A runner of twenty-six tying a shoe on a track"""


def scene_list(text: str) -> list[str]:
    scenes = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not scenes:
        raise ValueError("Give at least one scene, one per line.")
    return scenes


def join_wording(wording: str, scene: str) -> str:
    """The attribute goes FIRST. The text encoder reads left to right, so only the words after it can change."""
    wording = wording.strip()
    return f"{wording} {scene}" if wording else scene


# ---------------------------------------------------------------- the file

def new_steer(**info) -> dict:
    return {"text": None, "image": None, "sigmas": None, "info": dict(info, format=FORMAT)}


def to_tensors(steer: dict) -> tuple[dict, dict]:
    tensors = {}
    if steer.get("text") is not None:
        tensors["text"] = steer["text"].float().cpu().contiguous()
    if steer.get("image") is not None:
        tensors["image"] = steer["image"].float().cpu().contiguous()   # (noise levels, blocks, width)
        tensors["sigmas"] = steer["sigmas"].float().cpu().contiguous()  # (noise levels,), high to low
    if not tensors:
        raise ValueError("This direction is empty: fit a text part or an image part first.")
    return tensors, {"krea2_steer": json.dumps(steer.get("info", {}))}


def from_tensors(tensors: dict, metadata: dict | None) -> dict:
    steer = new_steer()
    steer["info"] = json.loads((metadata or {}).get("krea2_steer", "{}")) or {"format": FORMAT}
    steer["text"] = tensors.get("text")
    steer["image"] = tensors.get("image")
    steer["sigmas"] = tensors.get("sigmas")
    if steer["text"] is None and LORA_BIAS_KEY in tensors:  # a bias-only LoRA exported by this pack loads back as a text part
        steer["text"] = tensors[LORA_BIAS_KEY]
    return steer


def merge(base: dict | None, extra: dict) -> dict:
    """One direction can carry a text part and an image part. The newer part wins."""
    if base is None:
        return extra
    out = {"text": base.get("text"), "image": base.get("image"), "sigmas": base.get("sigmas"), "info": dict(base.get("info", {}))}
    if extra.get("text") is not None:
        out["text"] = extra["text"]
    if extra.get("image") is not None:
        out["image"], out["sigmas"] = extra["image"], extra["sigmas"]
    out["info"].update(extra.get("info", {}))
    return out


def describe(steer: dict) -> str:
    lines = []
    info = steer.get("info", {})
    if steer.get("text") is not None:
        lines.append(f"text part: {steer['text'].numel()} values, length {float(steer['text'].norm()):.2f}")
    if steer.get("image") is not None:
        levels, blocks, width = steer["image"].shape
        per_block = steer["image"].norm(dim=-1).mean(0)
        lines.append(f"image part: {levels} noise levels x {blocks} blocks x {width} values")
        lines.append("mean length per block: " + " ".join(f"{float(v):.1f}" for v in per_block))
        lines.append("noise levels: " + " ".join(f"{float(v):.3f}" for v in steer["sigmas"]))
    for key in sorted(info):
        lines.append(f"{key}: {info[key]}")
    return "\n".join(lines) or "empty direction"


# ---------------------------------------------------------------- text part

def aligned_difference(toward: torch.Tensor, away: torch.Tensor, keep: int) -> torch.Tensor:
    """Mean difference over the last ``keep`` slots: the scene's own words and the closing slots, which line up from the end."""
    keep = max(1, min(keep, toward.shape[0], away.shape[0]))
    return (toward[-keep:].float() - away[-keep:].float()).mean(0)


def mean_direction(parts: list[torch.Tensor]) -> tuple[torch.Tensor, float]:
    """The direction and how far the scenes agree on it (mean cosine to the average: 1 identical, 0 unrelated)."""
    stack = torch.stack(parts)
    vector = stack.mean(0)
    agreement = float(torch.nn.functional.cosine_similarity(stack, vector[None], dim=-1).mean())
    return vector, agreement


def text_tokens(diffusion_model, cond: torch.Tensor) -> torch.Tensor:
    """What the blocks read for one prompt: (slots, 6144). ``cond`` is the encoder output, (1, slots, 12*2560)."""
    weight = diffusion_model.txtmlp[3].weight
    with torch.no_grad():
        context = diffusion_model._unpack_context(cond.to(device=weight.device, dtype=weight.dtype if weight.dtype.is_floating_point else torch.float32))
        return diffusion_model.txtmlp(diffusion_model.txtfusion(context, mask=None, transformer_options={}))[0].float().cpu()


def text_patch(vector: torch.Tensor) -> dict:
    """The weight patch ComfyUI understands: add ``vector`` to the bias. Strength is given when the patch is added."""
    return {TEXT_BIAS_KEY: ("diff", (vector.float(),))}


# ---------------------------------------------------------------- image part

def sigma_of(timesteps) -> float:
    return float(timesteps.flatten()[0])


def image_token_range(transformer_options: dict, length: int) -> tuple[int, int]:
    start, end = transformer_options.get("img_slice", (0, length))
    end -= sum(transformer_options.get("reference_image_num_tokens", []) or [])  # reference pictures ride after the image tokens
    return start, end


class Recorder:
    """Collects the mean of the image tokens leaving every block, per noise level, over any number of model calls."""

    def __init__(self):
        self.sums: dict[float, torch.Tensor] = {}
        self.counts: dict[float, int] = {}

    def wrapper(self, executor, x, timesteps, *args, **kwargs):
        model = executor.class_obj
        sigma = round(sigma_of(timesteps), 5)
        means = [None] * len(model.blocks)

        def hook(index):
            def run(_module, _args, call_kwargs, output):
                start, end = image_token_range(call_kwargs.get("transformer_options", {}), output.shape[1])
                means[index] = output[:, start:end].float().mean(dim=(0, 1)).cpu()
            return run

        handles = [block.register_forward_hook(hook(i), with_kwargs=True) for i, block in enumerate(model.blocks)]
        try:
            return executor(x, timesteps, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
            if all(m is not None for m in means):
                stacked = torch.stack(means)
                self.sums[sigma] = self.sums.get(sigma, 0) + stacked
                self.counts[sigma] = self.counts.get(sigma, 0) + 1

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(sigmas high to low, means as (noise levels, blocks, width))."""
        if not self.sums:
            raise RuntimeError("Nothing was recorded: the model was never called.")
        sigmas = sorted(self.sums, reverse=True)
        return torch.tensor(sigmas), torch.stack([self.sums[s] / self.counts[s] for s in sigmas])


def image_difference(toward: Recorder, away: Recorder) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas_a, means_a = toward.result()
    sigmas_b, means_b = away.result()
    if sigmas_a.shape != sigmas_b.shape or not torch.allclose(sigmas_a, sigmas_b, atol=1e-4):
        raise RuntimeError("The two sides were recorded at different noise levels.")
    return sigmas_a, means_a - means_b


def vector_at(image: torch.Tensor, sigmas: torch.Tensor, sigma: float) -> torch.Tensor:
    """The per-block vectors at a noise level, linearly interpolated between the recorded ones. (blocks, width)"""
    if len(sigmas) == 1 or sigma >= float(sigmas[0]):
        return image[0]
    if sigma <= float(sigmas[-1]):
        return image[-1]
    for i in range(len(sigmas) - 1):
        high, low = float(sigmas[i]), float(sigmas[i + 1])
        if low <= sigma <= high:
            t = (high - sigma) / max(high - low, 1e-8)
            return image[i] * (1 - t) + image[i + 1] * t
    return image[-1]


def image_steerer(image: torch.Tensor, sigmas: torch.Tensor, strength: float, first_block: int, last_block: int,
                  sigma_high: float, sigma_low: float):
    """A model wrapper that adds ``strength * vector`` to the image tokens leaving blocks first..last while the noise
    level is between sigma_high and sigma_low. Text tokens and reference pictures are left alone."""
    image, sigmas = image.float(), sigmas.float()

    def wrapper(executor, x, timesteps, *args, **kwargs):
        sigma = sigma_of(timesteps)
        if strength == 0 or not (sigma_low - 1e-6 <= sigma <= sigma_high + 1e-6):
            return executor(x, timesteps, *args, **kwargs)
        model = executor.class_obj
        vectors = vector_at(image, sigmas, sigma) * strength
        first = max(first_block, 0)
        last = min(last_block, len(model.blocks) - 1, vectors.shape[0] - 1)

        def hook(index):
            # The stream is cumulative: what block i adds is still there at block i+1. So the first block in the range
            # gets the whole measured shift and each later one only the part that is new at that depth.
            step = vectors[index] if index == first else vectors[index] - vectors[index - 1]

            def run(_module, _args, call_kwargs, output):
                start, end = image_token_range(call_kwargs.get("transformer_options", {}), output.shape[1])
                output[:, start:end] += step.to(device=output.device, dtype=output.dtype)
                return output
            return run

        handles = [model.blocks[i].register_forward_hook(hook(i), with_kwargs=True) for i in range(first, last + 1)]
        try:
            return executor(x, timesteps, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

    return wrapper
