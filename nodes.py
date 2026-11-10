"""ComfyUI nodes for Krea 2 steering. The maths is in steer.py; this file only connects it to ComfyUI."""
import os

import torch
from safetensors.torch import save_file

import comfy.model_management
import comfy.patcher_extension
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths

from . import steer as S

FOLDER = "krea2_steer"
folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))
folder_paths.add_model_folder_path(FOLDER, os.path.join(os.path.dirname(__file__), "directions"))  # the ones shipped with the pack
CATEGORY = "Krea2/steer"


def check_model(model):
    diffusion_model = model.model.diffusion_model
    if not hasattr(diffusion_model, "txtfusion") or not hasattr(diffusion_model, "txtmlp"):
        raise ValueError("These nodes need a Krea 2 model (they patch its txtmlp bias and its transformer blocks).")
    return diffusion_model


def encode(clip, text):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def save(steer_data, name):
    if not name.strip():
        return ""
    folder = folder_paths.get_folder_paths(FOLDER)[0]
    path = os.path.join(folder, name.strip().removesuffix(".safetensors") + ".safetensors")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensors, metadata = S.to_tensors(steer_data)
    save_file(tensors, path, metadata=metadata)
    return path


class Krea2SteerFitText:
    """Text part of a direction: how the text tokens the model reads shift between two wordings of the same scenes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "toward": ("STRING", {"multiline": True, "default": "", "tooltip": "Wording of what positive strength moves toward. Put the look in words, as one or two sentences."}),
            "away": ("STRING", {"multiline": True, "default": "", "tooltip": "Wording of the opposite. Leave empty to compare against the scene alone."}),
            "scenes": ("STRING", {"multiline": True, "default": S.DEFAULT_SCENES, "tooltip": "Everyday scenes, one per line. The direction is what stays the same across all of them."}),
            "save_as": ("STRING", {"default": "", "tooltip": "File name under models/krea2_steer. Empty = do not save."}),
        }, "optional": {"steer": ("KREA2_STEER", {"tooltip": "An existing direction to add this text part to."})}}

    RETURN_TYPES = ("KREA2_STEER", "STRING")
    RETURN_NAMES = ("steer", "report")
    FUNCTION = "fit"
    CATEGORY = CATEGORY

    def fit(self, model, clip, toward, away, scenes, save_as, steer=None):
        diffusion_model = check_model(model)
        if not toward.strip():
            raise ValueError("'toward' is empty: say in words what the direction is.")
        comfy.model_management.load_models_gpu([model])
        scenes = S.scene_list(scenes)
        parts, lengths = [], []
        progress = comfy.utils.ProgressBar(len(scenes))
        for scene in scenes:
            tokens = lambda text: S.text_tokens(diffusion_model, encode(clip, text)[0][0])  # noqa: E731
            plain = tokens(scene)
            keep = plain.shape[0] - 2  # the first word of the scene tokenises differently after a sentence; drop it
            a = tokens(S.join_wording(toward, scene))
            b = tokens(S.join_wording(away, scene)) if away.strip() else plain
            parts.append(S.aligned_difference(a, b, keep))
            lengths.append(float(plain.norm(dim=-1).mean()))
            progress.update(1)
        vector, agreement = S.mean_direction(parts)
        typical = sum(lengths) / len(lengths)
        info = {"toward": toward.strip(), "away": away.strip(), "scenes": len(scenes), "text_agreement": round(agreement, 3),
                "text_relative_length": round(float(vector.norm()) / typical, 4)}
        out = S.merge(steer, dict(S.new_steer(**info), text=vector))
        path = save(out, save_as)
        return (out, S.describe(out) + (f"\nsaved: {path}" if path else ""))


class Krea2SteerFitImage:
    """Image part of a direction, from two wordings: the model renders the same scenes and seeds both ways and the
    mean of the image tokens leaving every block is compared, per noise level."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "toward": ("STRING", {"multiline": True, "default": ""}),
            "away": ("STRING", {"multiline": True, "default": "", "tooltip": "Leave empty to compare against the scene alone."}),
            "scenes": ("STRING", {"multiline": True, "default": S.DEFAULT_SCENES}),
            "width": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 100, "tooltip": "Use the step count you render with: the direction is stored per noise level."}),
            "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "euler"}),
            "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 2**32 - 1}),
            "save_as": ("STRING", {"default": ""}),
        }, "optional": {"steer": ("KREA2_STEER",)}}

    RETURN_TYPES = ("KREA2_STEER", "STRING")
    RETURN_NAMES = ("steer", "report")
    FUNCTION = "fit"
    CATEGORY = CATEGORY

    def fit(self, model, clip, toward, away, scenes, width, height, steps, sampler_name, scheduler, seed, save_as, steer=None):
        check_model(model)
        if not toward.strip():
            raise ValueError("'toward' is empty: say in words what the direction is.")
        scenes = S.scene_list(scenes)
        latent = torch.zeros([1, 16, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        recorders = (S.Recorder(), S.Recorder())
        progress = comfy.utils.ProgressBar(len(scenes) * 2)
        for index, scene in enumerate(scenes):
            noise = comfy.sample.prepare_noise(latent, seed + index)  # the same noise for both wordings
            for recorder, wording in zip(recorders, (toward, away)):
                recording = model.clone()
                recording.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, S.WRAPPER_KEY + "_record", recorder.wrapper)
                cond = encode(clip, S.join_wording(wording, scene))
                comfy.sample.sample(recording, noise, steps, 1.0, sampler_name, scheduler, cond, cond, latent, disable_pbar=True, seed=seed + index)
                progress.update(1)
        sigmas, image = S.image_difference(*recorders)
        info = {"toward": toward.strip(), "away": away.strip(), "image_scenes": len(scenes), "image_size": f"{width}x{height}",
                "image_steps": steps, "image_source": "wordings"}
        out = S.merge(steer, dict(S.new_steer(**info), image=image, sigmas=sigmas))
        path = save(out, save_as)
        return (out, S.describe(out) + (f"\nsaved: {path}" if path else ""))


class Krea2SteerFitPictures:
    """Image part of a direction, from two sets of pictures and no words: for example real photographs against the
    model's own renders. Each picture is noised to every level of the schedule and passed through the model once."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "toward": ("IMAGE", {"tooltip": "Pictures of what positive strength moves toward (all one size)."}),
            "away": ("IMAGE", {"tooltip": "Pictures of what it moves away from (the same size)."}),
            "caption": ("STRING", {"multiline": True, "default": "A photograph.", "tooltip": "The same neutral prompt is used for every picture on both sides."}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
            "seed": ("INT", {"default": 1, "min": 0, "max": 2**32 - 1}),
            "save_as": ("STRING", {"default": ""}),
        }, "optional": {"steer": ("KREA2_STEER",)}}

    RETURN_TYPES = ("KREA2_STEER", "STRING")
    RETURN_NAMES = ("steer", "report")
    FUNCTION = "fit"
    CATEGORY = CATEGORY

    def fit(self, model, clip, vae, toward, away, caption, steps, scheduler, seed, save_as, steer=None):
        check_model(model)
        cond = encode(clip, caption)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)
        sampler = comfy.samplers.sampler_object("euler")
        recorders = (S.Recorder(), S.Recorder())
        progress = comfy.utils.ProgressBar((toward.shape[0] + away.shape[0]) * steps)
        for recorder, pictures in zip(recorders, (toward, away)):
            recording = model.clone()
            recording.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, S.WRAPPER_KEY + "_record", recorder.wrapper)
            for index in range(pictures.shape[0]):
                latent = vae.encode(pictures[index:index + 1, :, :, :3])
                noise = comfy.sample.prepare_noise(latent, seed + index)  # picture i on both sides gets the same noise
                for level in range(steps):
                    # One model call on the picture noised to this level: a one-step schedule from sigma to zero.
                    pair = torch.stack([sigmas[level], sigmas[level] * 0])
                    comfy.sample.sample_custom(recording, noise, 1.0, sampler, pair, cond, cond, latent, disable_pbar=True, seed=seed + index)
                    progress.update(1)
        levels, image = S.image_difference(*recorders)
        info = {"image_source": "pictures", "image_pictures": f"{toward.shape[0]} toward, {away.shape[0]} away",
                "image_size": f"{toward.shape[2]}x{toward.shape[1]}", "image_steps": steps, "caption": caption.strip()}
        out = S.merge(steer, dict(S.new_steer(**info), image=image, sigmas=levels))
        path = save(out, save_as)
        return (out, S.describe(out) + (f"\nsaved: {path}" if path else ""))


class Krea2SteerApply:
    """Steers a Krea 2 model along a direction. Negative strength moves away from it (a negative prompt at CFG 1)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "steer": ("KREA2_STEER",),
            "text_strength": ("FLOAT", {"default": 0.0, "min": -64.0, "max": 64.0, "step": 0.25, "tooltip": "Text part. 1 = the measured difference between the two wordings; useful range about -8 to 8."}),
            "image_strength": ("FLOAT", {"default": 0.0, "min": -64.0, "max": 64.0, "step": 0.25, "tooltip": "Image part. 1 = the measured difference inside the blocks."}),
            "first_block": ("INT", {"default": 0, "min": 0, "max": 27}),
            "last_block": ("INT", {"default": 27, "min": 0, "max": 27}),
            "start_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Image part acts from this noise level (1 = pure noise, the first step)..."}),
            "end_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "...down to this one (0 = the finished picture). Low levels are texture, high levels are layout."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model, steer, text_strength, image_strength, first_block, last_block, start_sigma, end_sigma):
        check_model(model)
        out = model.clone()
        if text_strength != 0:
            if steer.get("text") is None:
                raise ValueError("This direction has no text part: fit one with 'Krea2 Steer Fit (text)' or set text_strength to 0.")
            out.add_patches(S.text_patch(steer["text"]), text_strength)
        if image_strength != 0:
            if steer.get("image") is None:
                raise ValueError("This direction has no image part: fit one with 'Krea2 Steer Fit (image)' or set image_strength to 0.")
            out.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, S.WRAPPER_KEY,
                                     S.image_steerer(steer["image"], steer["sigmas"], image_strength, first_block, last_block, start_sigma, end_sigma))
        return (out,)


class Krea2SteerLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"name": (folder_paths.get_filename_list(FOLDER),)}}

    RETURN_TYPES = ("KREA2_STEER", "STRING")
    RETURN_NAMES = ("steer", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, name):
        tensors, metadata = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise(FOLDER, name), safe_load=True, return_metadata=True)
        out = S.from_tensors(tensors, metadata)
        return (out, S.describe(out))


class Krea2SteerExportLoRA:
    """Writes the text part as a bias-only LoRA (one tensor), usable in any ComfyUI with a plain LoRA loader and no
    custom nodes. The image part cannot be a LoRA: it acts while sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"steer": ("KREA2_STEER",), "filename": ("STRING", {"default": "krea2-steer/direction"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def export(self, steer, filename):
        if steer.get("text") is None:
            raise ValueError("This direction has no text part to export.")
        path = os.path.join(folder_paths.get_folder_paths("loras")[0], filename.strip().removesuffix(".safetensors") + ".safetensors")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _, metadata = S.to_tensors(steer)
        save_file({S.LORA_BIAS_KEY: steer["text"].float().cpu().contiguous()}, path, metadata=metadata)
        return (path,)


NODE_CLASS_MAPPINGS = {
    "Krea2SteerFitText": Krea2SteerFitText,
    "Krea2SteerFitImage": Krea2SteerFitImage,
    "Krea2SteerFitPictures": Krea2SteerFitPictures,
    "Krea2SteerApply": Krea2SteerApply,
    "Krea2SteerLoad": Krea2SteerLoad,
    "Krea2SteerExportLoRA": Krea2SteerExportLoRA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2SteerFitText": "Krea2 Steer Fit (text, from two wordings)",
    "Krea2SteerFitImage": "Krea2 Steer Fit (image, from two wordings)",
    "Krea2SteerFitPictures": "Krea2 Steer Fit (image, from two picture sets)",
    "Krea2SteerApply": "Krea2 Steer Apply",
    "Krea2SteerLoad": "Krea2 Steer Load",
    "Krea2SteerExportLoRA": "Krea2 Steer Export LoRA (text part)",
}
