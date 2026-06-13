import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aggregation import aggregate_probabilities


def test_mean_aggregation_preserves_average_probabilities():
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )

    aggregated = aggregate_probabilities(probabilities)

    assert np.allclose(aggregated, [0.4, 0.5, 0.1])


def test_top_confidence_aggregation_uses_most_confident_windows():
    probabilities = np.array(
        [
            [0.45, 0.40, 0.15],
            [0.05, 0.90, 0.05],
            [0.10, 0.20, 0.70],
            [0.80, 0.10, 0.10],
        ],
        dtype=np.float32,
    )

    aggregated = aggregate_probabilities(
        probabilities,
        mode="top_confidence_mean",
        top_fraction=0.25,
    )

    assert int(np.argmax(aggregated)) == 1
    assert np.isclose(np.sum(aggregated), 1.0)


def test_unknown_aggregation_mode_is_rejected():
    probabilities = np.array([[0.5, 0.5]], dtype=np.float32)

    with pytest.raises(ValueError, match="Unknown aggregation mode"):
        aggregate_probabilities(probabilities, mode="unknown")
