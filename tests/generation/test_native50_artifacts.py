import numpy as np
import pytest

from omg.benchmarks.artifacts import GENERATED_KEYS, verify_schema
from omg.benchmarks.runners.artifact import _validate_generated


@pytest.mark.parametrize('fps', [25.0, 60.0, float('nan')])
def test_benchmark_artifact_readers_reject_non_native_rates(fps):
    arrays = {
        'qpos_36': np.zeros((1, 100, 36), dtype=np.float32),
        'fps': np.array([fps]),
        'captions': np.array(['walk']),
        'dataset': np.array(['test']),
        'dataset_index': np.array([0]),
    }
    with pytest.raises(ValueError, match='50Hz'):
        verify_schema(arrays, GENERATED_KEYS, label='generated')
    with pytest.raises(ValueError, match='50Hz'):
        _validate_generated(arrays)
    arrays['fps'][:] = 50.0
    verify_schema(arrays, GENERATED_KEYS, label='generated')
    _validate_generated(arrays)
