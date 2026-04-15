import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocess import drop_nulls, encode_labels, scale_features


def test_drop_nulls():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    result = drop_nulls(df)
    assert len(result) == 2


def test_encode_labels():
    df = pd.DataFrame({"label": ["cat", "dog", "cat"]})
    result = encode_labels(df, ["label"])
    assert result["label"].dtype in ["int32", "int64"]


def test_scale_features():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    result, scaler = scale_features(df, ["x", "y"])
    assert abs(result["x"].mean()) < 1e-10
