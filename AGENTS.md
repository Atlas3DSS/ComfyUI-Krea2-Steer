# For agents

This pack adds steering directions to Krea 2 in ComfyUI. Read `README.md` for the idea; this file is the contract.

## Layout

| File | Holds |
| --- | --- |
| `steer.py` | All maths and model hooks. Imports only torch. Change behaviour here. |
| `nodes.py` | ComfyUI nodes: inputs, model loading, saving. No maths. |
| `tools/lab.py` | Fixed-seed comparison sheets over ComfyUI's HTTP API. |
| `examples/` | Plans for `lab.py` and one API-format workflow. |
| `directions/` | Ready-made directions for stock Krea 2 Turbo; `Krea2SteerLoad` lists them next to `models/krea2_steer`. |
| `docs/` | The comparison images used in the README. |
| `tests/test_steer.py` | Unit tests on a toy model. `python -m pytest`. |

## Node contract (API format)

Type `KREA2_STEER` is a dict: `text` (6144,) or None, `image` (levels, blocks, 6144) or None, `sigmas` (levels,) high
to low, `info` (dict of plain values). The API does not fill defaults: send every required input.

| class_type | Required inputs | Optional | Outputs |
| --- | --- | --- | --- |
| `Krea2SteerFitText` | `model`, `clip`, `toward`, `away` (may be ""), `scenes` (one per line), `save_as` ("" = no file) | `steer` (merge into) | `steer`, `report` |
| `Krea2SteerFitImage` | the same plus `width`, `height`, `steps`, `sampler_name`, `scheduler`, `seed` | `steer` | `steer`, `report` |
| `Krea2SteerFitPictures` | `model`, `clip`, `vae`, `toward` (IMAGE batch), `away` (IMAGE batch), `caption`, `steps`, `scheduler`, `seed`, `save_as` | `steer` | `steer`, `report` |
| `Krea2SteerApply` | `model`, `steer`, `text_strength`, `image_strength`, `first_block`, `last_block` (0-27), `start_sigma`, `end_sigma` (1 = noise, 0 = done) | | `model` |
| `Krea2SteerLoad` | `name` (file under `models/krea2_steer`, e.g. `lab/smooth.safetensors`) | | `steer`, `report` |
| `Krea2SteerExportLoRA` | `steer`, `filename` (under `models/loras`) | | `path` |

File format: safetensors with tensors `text`, `image`, `sigmas` and metadata key `krea2_steer` (JSON of `info`).
The exported LoRA holds one tensor, `diffusion_model.txtmlp.3.diff_b`.

## Rules that keep results honest

1. Compare with the same seed, prompt, size, steps and sampler. Change one thing per arm.
2. Fit the image part with the step count and scheduler you render with.
3. Start small: image strength 0.25, text strength 2. Sweep before you conclude a direction does nothing or is broken.
4. Look at full-size crops, not only thumbnails. The detail number rises with noise too.
5. A direction is tied to the weights it was fitted on. Refit for another checkpoint and say so in the file name.
6. Do not print prompts in logs. `lab.py` prints arm and test names only.

## Adding another model

Everything Krea-specific is in three places in `steer.py`: `TEXT_BIAS_KEY` (the last text layer's bias),
`text_tokens` (how encoder output becomes the tokens the blocks read) and the `blocks` list the hooks attach to, with
`image_token_range` saying which tokens are image. Port those three and the nodes work unchanged.
