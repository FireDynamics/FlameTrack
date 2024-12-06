import pytest
import numpy as np
from ir_reader.analysis.IR_analysis import get_dewarp_parameters

def test_get_dewarp_parameters_valid():
    corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
    target_pixels_width = 200
    target_pixels_height = 200
    result = get_dewarp_parameters(corners, target_pixels_width, target_pixels_height)
    assert result is not None  # Ensure function returns something valid
    assert isinstance(result, dict)  # Ensure the result is a dictionary
    assert 'transformation_matrix' in result  # Ensure the key exists
    assert 'target_pixels_width' in result  # Ensure the key exists
    assert 'target_pixels_height' in result  # Ensure the key exists
    assert 'target_ratio' in result  # Ensure the key exists
    assert result['target_pixels_width'] == target_pixels_width  # Ensure width matches input
    assert result['target_pixels_height'] == target_pixels_height  # Ensure height matches input
    assert result['target_ratio'] == target_pixels_height / target_pixels_width  # Ensure ratio is correct

def test_get_dewarp_parameters_target_ratio():
    corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
    target_ratio = 1.0
    result = get_dewarp_parameters(corners, target_ratio=target_ratio)
    assert result is not None  # Ensure function returns something valid
    assert isinstance(result, dict)  # Ensure the result is a dictionary
    assert 'transformation_matrix' in result  # Ensure the key exists
    assert 'target_pixels_width' in result  # Ensure the key exists
    assert 'target_pixels_height' in result  # Ensure the key exists
    assert 'target_ratio' in result  # Ensure the key exists
    assert result['target_pixels_width'] == result['target_pixels_height']  # Width and height should be equal for ratio 1.0
    assert result['target_ratio'] == 1.0  # Ensure ratio matches the input ratio

def test_get_dewarp_parameters_missing_arguments():
    corners = [[0, 0], [100, 0], [100, 100], [0, 100]]
    with pytest.raises(ValueError, match='Either target_pixels_width and target_pixels_height or target_ratio must be provided'):
        get_dewarp_parameters(corners)
