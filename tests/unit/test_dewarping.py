import pytest

from flametrack.processing.dewarping import (
    dewarp_lateral_flame_spread,
    dewarp_room_corner_remap,
)


@pytest.fixture
def dummy_experiment():
    class Dummy:
        def get_data(self, datatype):
            return None

    return Dummy()


def test_dewarp_room_corner_remap_raises_on_wrong_number_of_points(dummy_experiment):
    points = [(0, 0), (1, 0), (2, 0)]  # Only 3 instead of 6

    with pytest.raises(ValueError, match="Expected exactly 6 points"):
        list(
            dewarp_room_corner_remap(
                experiment=dummy_experiment,
                points=points,
                target_ratio=1.0,
                target_pixels_width=32,
                target_pixels_height=32,
                testing=True,
            )
        )


def test_dewarp_room_corner_remap_raises_on_too_small_image_size(dummy_experiment):
    points = [(0, 0), (10, 0), (20, 0), (20, 10), (10, 10), (0, 10)]

    with pytest.raises(ValueError, match="Target image size too small"):
        list(
            dewarp_room_corner_remap(
                experiment=dummy_experiment,
                points=points,
                target_ratio=1.0,
                target_pixels_width=5,  # invalid (<=10)
                target_pixels_height=5,  # invalid (<=10)
                testing=True,
            )
        )


def test_dewarp_lfs_raises_on_too_small_image_size(dummy_experiment):
    points = [(0, 0), (20, 0), (20, 10), (0, 10)]

    with pytest.raises(ValueError, match="target_pixels_width must be greater than 10"):
        list(
            dewarp_lateral_flame_spread(
                experiment=dummy_experiment,
                points=points,
                target_ratio=1.0,
                target_pixels_width=0,  # invalid
                target_pixels_height=100,
                testing=True,
            )
        )

    with pytest.raises(
        ValueError, match="target_pixels_height must be greater than 10"
    ):
        list(
            dewarp_lateral_flame_spread(
                experiment=dummy_experiment,
                points=points,
                target_ratio=1.0,
                target_pixels_width=100,
                target_pixels_height=5,  # invalid
                testing=True,
            )
        )


def test_dewarp_lfs_raises_on_invalid_number_of_points(dummy_experiment):
    bad_points = [(0, 0), (10, 0), (20, 0)]  # Only 3 instead of 4

    with pytest.raises(ValueError, match="Exactly 4 corner points are required"):
        list(
            dewarp_lateral_flame_spread(
                experiment=dummy_experiment,
                points=bad_points,
                target_ratio=1.0,
                target_pixels_width=100,
                target_pixels_height=100,
                testing=True,
            )
        )
