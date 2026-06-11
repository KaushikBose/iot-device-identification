import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import NUM_CLASSES
from model import build_cnn


def test_build_cnn_default_pooling():
    input_shape = (257, 61, 1)
    model = build_cnn(input_shape, NUM_CLASSES)  # should default to "avg"

    assert model is not None
    assert model.output_shape == (None, NUM_CLASSES)
    assert model.loss == "categorical_crossentropy"
    assert model.layers[-1].activation.__name__ == "softmax"
    
    # Assert that the pooling layer is GlobalAveragePooling2D
    pooling_layers = [l for l in model.layers if "global_average_pooling" in l.name]
    assert len(pooling_layers) == 1


def test_build_cnn_pooling_modes():
    input_shape = (257, 61, 1)
    model_avg = build_cnn(input_shape, NUM_CLASSES, pooling="avg")
    model_max = build_cnn(input_shape, NUM_CLASSES, pooling="max")
    model_avgmax = build_cnn(input_shape, NUM_CLASSES, pooling="avgmax")
    model_flat = build_cnn(input_shape, NUM_CLASSES, pooling="flatten")

    assert model_avg.output_shape == (None, NUM_CLASSES)
    assert model_max.output_shape == (None, NUM_CLASSES)
    assert model_avgmax.output_shape == (None, NUM_CLASSES)
    assert model_flat.output_shape == (None, NUM_CLASSES)

    # Check for respective pooling layers
    assert any("global_average_pooling" in l.name for l in model_avg.layers)
    assert any("global_max_pooling" in l.name for l in model_max.layers)
    assert any("concatenate" in l.name for l in model_avgmax.layers)
    assert any("flatten" in l.name for l in model_flat.layers)

    # Parameter size comparisons
    params_avg = model_avg.count_params()
    params_flat = model_flat.count_params()

    # The flat model should have ~15 million parameters, avg should have ~450k
    assert params_flat > 14_000_000
    assert params_avg < 500_000
    assert params_flat > params_avg * 30
