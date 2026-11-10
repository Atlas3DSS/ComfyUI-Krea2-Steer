"""The maths and the hooks, on a toy model. Needs only torch:  python -m pytest tests"""
import importlib.util
import pathlib

import torch

spec = importlib.util.spec_from_file_location("steer", pathlib.Path(__file__).resolve().parents[1] / "steer.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


class Block(torch.nn.Module):
    def forward(self, x, transformer_options=None):
        return x + 1.0


class Toy:
    """Three blocks over [2 text tokens | 4 image tokens | 1 reference token]."""

    def __init__(self):
        self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])

    def __call__(self, x, timesteps):
        options = {"img_slice": [2, 7], "reference_image_num_tokens": [1]}
        for block in self.blocks:
            x = block(x, transformer_options=options)
        return x


class Executor:
    def __init__(self, model):
        self.class_obj = model

    def __call__(self, x, timesteps, *args, **kwargs):
        return self.class_obj(x, timesteps)


def run(wrapper, sigma=0.5):
    return wrapper(Executor(Toy()), torch.zeros(1, 7, 4), torch.tensor([sigma]))


def test_scenes_and_wording_order():
    assert S.scene_list("a\n\n# note\n b ") == ["a", "b"]
    assert S.join_wording("Soft light.", "A cat") == "Soft light. A cat"
    assert S.join_wording("  ", "A cat") == "A cat"


def test_aligned_difference_uses_the_last_slots():
    toward = torch.cat([torch.full((3, 2), 9.0), torch.ones(4, 2)])  # three extra leading slots are ignored
    away = torch.zeros(4, 2)
    assert torch.allclose(S.aligned_difference(toward, away, 4), torch.ones(2))


def test_mean_direction_agreement():
    vector, agreement = S.mean_direction([torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])])
    assert torch.allclose(vector, torch.tensor([1.0, 0.0])) and abs(agreement - 1) < 1e-6
    _, mixed = S.mean_direction([torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])])
    assert mixed < 0.8


def test_recorder_sees_only_image_tokens_per_block():
    recorder = S.Recorder()
    run(recorder.wrapper)
    sigmas, means = recorder.result()
    assert sigmas.tolist() == [0.5] and means.shape == (1, 3, 4)
    assert torch.allclose(means[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_steerer_adds_the_measured_shift_once_and_spares_text_and_reference():
    image = torch.tensor([[[1.0] * 4, [3.0] * 4, [6.0] * 4]])  # one noise level, three blocks
    out = run(S.image_steerer(image, torch.tensor([0.5]), 1.0, 0, 2, 1.0, 0.0))
    assert torch.allclose(out[0, 2:6], torch.full((4, 4), 3.0 + 6.0))  # the last block's shift, not 1+3+6
    assert torch.allclose(out[0, :2], torch.full((2, 4), 3.0)) and torch.allclose(out[0, 6], torch.full((4,), 3.0))


def test_steerer_respects_strength_blocks_and_noise_window():
    image = torch.tensor([[[1.0] * 4, [3.0] * 4, [6.0] * 4]])
    sigmas = torch.tensor([0.5])
    assert torch.allclose(run(S.image_steerer(image, sigmas, -2.0, 1, 1, 1.0, 0.0))[0, 2:6], torch.full((4, 4), 3.0 - 6.0))
    assert torch.allclose(run(S.image_steerer(image, sigmas, 1.0, 0, 2, 0.4, 0.0), sigma=0.5)[0, 2:6], torch.full((4, 4), 3.0))


def test_vector_at_interpolates_between_noise_levels():
    image = torch.tensor([[[0.0]], [[10.0]]])
    sigmas = torch.tensor([1.0, 0.0])
    assert float(S.vector_at(image, sigmas, 0.25)) == 7.5
    assert float(S.vector_at(image, sigmas, 2.0)) == 0.0 and float(S.vector_at(image, sigmas, -1.0)) == 10.0


def test_file_round_trip_and_merge():
    text = dict(S.new_steer(toward="a"), text=torch.ones(6))
    image = dict(S.new_steer(image_steps=8), image=torch.ones(2, 3, 6), sigmas=torch.tensor([1.0, 0.5]))
    both = S.merge(text, image)
    tensors, metadata = S.to_tensors(both)
    back = S.from_tensors(tensors, metadata)
    assert back["text"].shape == (6,) and back["image"].shape == (2, 3, 6)
    assert back["info"]["toward"] == "a" and back["info"]["image_steps"] == 8
    assert "text part" in S.describe(back) and "image part" in S.describe(back)
    assert S.from_tensors({S.LORA_BIAS_KEY: torch.ones(6)}, None)["text"].shape == (6,)
