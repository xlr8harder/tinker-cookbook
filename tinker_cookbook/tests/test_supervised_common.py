import pytest
import torch
import tinker

from tinker_cookbook.supervised.common import datum_from_model_input_weights


def _get_weights(datum: tinker.Datum) -> list[float]:
    return datum.loss_fn_inputs["weights"].data


def test_normalize_weights_enabled():
    model_input = tinker.ModelInput.from_ints([1, 2, 3, 4])
    weights = torch.tensor([0.0, 1.0, 1.0, 2.0], dtype=torch.float32)
    datum = datum_from_model_input_weights(model_input, weights, normalize_weights=True)
    assert _get_weights(datum) == pytest.approx([0.25, 0.25, 0.5])


def test_normalize_weights_disabled():
    model_input = tinker.ModelInput.from_ints([1, 2, 3, 4])
    weights = torch.tensor([0.0, 1.0, 1.0, 2.0], dtype=torch.float32)
    datum = datum_from_model_input_weights(model_input, weights, normalize_weights=False)
    assert _get_weights(datum) == [1.0, 1.0, 2.0]


def test_normalize_weights_all_zero():
    model_input = tinker.ModelInput.from_ints([1, 2, 3, 4])
    weights = torch.zeros(4, dtype=torch.float32)
    datum = datum_from_model_input_weights(model_input, weights, normalize_weights=True)
    assert _get_weights(datum) == [0.0, 0.0, 0.0]
