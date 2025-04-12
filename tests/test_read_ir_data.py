import pytest
import numpy as np
import os

from flametrack.analysis.IR_analysis import read_IR_data

def test_read_IR_data_valid():
    file_path = os.path.join(os.path.dirname(__file__), "fixtures", "test_data.txt")
    result = read_IR_data(file_path)

def test_read_IR_data_valid():
    # Simulate a valid IR data file
    ir_data = """24,37;24,73;24,79;24,21;
    24,41;24,57;24,27;24,07;
    24,25;24,37;24,23;24,67;"""
    file_path = os.path.join(os.path.dirname(__file__), "fixtures", "test_data.csv")
    result = read_IR_data(file_path)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 4)
    # Check specific values
    expected_values = np.array([
        [24.37, 24.73, 24.79, 24.21],
        [24.41, 24.57, 24.27, 24.07],
        [24.25, 24.37, 24.23, 24.67]
    ])
    np.testing.assert_almost_equal(result, expected_values)


def test_read_IR_data_no_data():
    # Assuming an invalid IR data file exists at 'empty_data.txt'
    file_path = os.path.join(os.path.dirname(__file__), "fixtures", "empty_data.csv")
    with pytest.raises(ValueError, match='No data found in file, check file format!'):
        read_IR_data(file_path)