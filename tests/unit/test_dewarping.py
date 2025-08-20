import pytest

from flametrack.processing.dewarping import (
    DewarpConfig,
    dewarp_lateral_flame_spread,
    dewarp_room_corner_remap,
)


@pytest.fixture
def dummy_experiment():
    class Dummy:
        def get_data(self, _):
            return None

    return Dummy()


def test_dewarp_room_corner_remap_raises_on_wrong_number_of_points(dummy_experiment):
    points = [(0, 0), (1, 0), (2, 0)]  # nur 3 statt 6

    cfg = DewarpConfig(
        target_ratio=1.0,
        target_pixels_width=32,
        target_pixels_height=32,
        testing=True,
    )

    with pytest.raises(ValueError, match="Expected exactly 6 points"):
        list(dewarp_room_corner_remap(dummy_experiment, points, cfg))


def test_dewarp_room_corner_remap_raises_on_too_small_image_size(dummy_experiment):
    points = [(0, 0), (10, 0), (20, 0), (20, 10), (10, 10), (0, 10)]

    cfg = DewarpConfig(
        target_ratio=1.0,
        target_pixels_width=5,  # invalid (<=10)
        target_pixels_height=5,  # invalid (<=10)
        testing=True,
    )

    with pytest.raises(ValueError, match="Target image size too small"):
        list(dewarp_room_corner_remap(dummy_experiment, points, cfg))


def test_dewarp_lfs_raises_on_too_small_image_size(dummy_experiment):
    points = [(0, 0), (20, 0), (20, 10), (0, 10)]

    cfg_w = DewarpConfig(
        target_ratio=1.0,
        target_pixels_width=0,  # invalid
        target_pixels_height=100,
        testing=True,
    )
    with pytest.raises(ValueError, match="target_pixels_width must be greater than 10"):
        list(dewarp_lateral_flame_spread(dummy_experiment, points, cfg_w))

    cfg_h = DewarpConfig(
        target_ratio=1.0,
        target_pixels_width=100,
        target_pixels_height=5,  # invalid
        testing=True,
    )
    with pytest.raises(
        ValueError, match="target_pixels_height must be greater than 10"
    ):
        list(dewarp_lateral_flame_spread(dummy_experiment, points, cfg_h))


def test_dewarp_lfs_raises_on_invalid_number_of_points(dummy_experiment):
    bad_points = [(0, 0), (10, 0), (20, 0)]  # nur 3 statt 4

    cfg = DewarpConfig(
        target_ratio=1.0,
        target_pixels_width=100,
        target_pixels_height=100,
        testing=True,
    )

    with pytest.raises(ValueError, match="Exactly 4 corner points are required"):
        list(dewarp_lateral_flame_spread(dummy_experiment, bad_points, cfg))
