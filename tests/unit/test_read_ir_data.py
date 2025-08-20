import numpy as np
import pytest

from flametrack.analysis.ir_analysis import read_ir_data


def test_read_IR_data_valid(fixture_path):
    file_path = fixture_path / "test_data.csv"
    assert file_path.exists(), f"{file_path} does not exist"

    result = read_ir_data(str(file_path))

    expected_values = np.array(
        [
            [24.37, 24.73, 24.79, 24.21],
            [24.41, 24.57, 24.27, 24.07],
            [24.25, 24.37, 24.23, 24.67],
        ]
    )
    np.testing.assert_almost_equal(result, expected_values)


def test_read_IR_data_no_data(fixture_path):
    file_path = fixture_path / "empty_data.csv"
    assert file_path.exists(), f"{file_path} does not exist"

    with pytest.raises(ValueError, match="No data found in file, check file format!"):
        read_ir_data(str(file_path))
