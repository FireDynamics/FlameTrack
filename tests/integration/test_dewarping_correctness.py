"""
Automated correctness tests for the LFS and Room Corner dewarping pipelines.

For each of the 4 rotation indices (0 = upright, 1 = 90° CW, 2 = 180°, 3 = 270° CW)
we:
  1. Create a synthetic scene with a known perspective distortion.
  2. Simulate a camera mounted at that angle (np.rot90).
  3. Run the production dewarping code (same path as the GUI).
  4. Assert that the dewarped output is numerically close to the undistorted original.

Both experiments are covered:
  - LFS  : single plate, letter F, dewarp_lateral_flame_spread()
  - RCE  : two plates F + E,  dewarp_room_corner_remap()
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import h5py
import numpy as np
import pytest
from numpy.typing import NDArray

# ── production code under test ────────────────────────────────────────────────
from flametrack.processing.dewarping import (
    DewarpConfig,
    dewarp_lateral_flame_spread,
    dewarp_room_corner_remap,
)

# ── thresholds ────────────────────────────────────────────────────────────────
# Images are float32 in [0, 1].  After dewarping with mild perspective distortion
# the RMSE vs. the clean original is dominated by bilinear interpolation
# artifacts on the sharp (binary) edges of the F / E letters — empirically ≈ 0.05.
# A threshold of 0.09 gives comfortable headroom above this noise floor while
# still catching any systematic mis-orientation of the dewarped plate (which
# produces RMSE > 0.15 even for mild distortions).
RMSE_THRESHOLD = 0.09


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic scenes
# ─────────────────────────────────────────────────────────────────────────────


def _make_f(size: int = 200) -> NDArray[np.float32]:
    """Letter F as a float32 image in [0, 1]."""
    img = np.zeros((size, size), dtype=np.float32)
    s = size / 200
    bw = max(1, int(30 * s))
    img[int(30 * s) : int(170 * s), int(20 * s) : int(20 * s) + bw] = 1.0  # vertical
    img[int(30 * s) : int(30 * s) + bw, int(20 * s) : int(150 * s)] = 1.0  # top bar
    img[int(100 * s) : int(100 * s) + bw, int(20 * s) : int(120 * s)] = 1.0  # mid bar
    return img


def _make_fe(size: int = 200) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Two-plate scene (F left, E right) on a (size × 2*size) canvas.

    The two plates are adjacent (share an inner edge at x = int(180*s)) so
    that the 6 corner points form two non-overlapping 160×160 plate regions:
      left  plate: x in [lx0, lx1],  pts6[[0,1,4,5]]
      right plate: x in [lx1, rx1],  pts6[[1,2,3,4]]

    Returns
    -------
    canvas   : float32 image of shape (size, 2*size)
    pts6     : 6 corner points [p0..p5] in canvas coordinates (before distortion)
    """
    h, w = size, 2 * size
    canvas = np.zeros((h, w), dtype=np.float32)
    s = size / 200
    bw = max(1, int(20 * s))

    # Both plates span the same vertical band (20..180) and each is 160×160.
    # They share the inner edge at x = lx1 = rx0 = int(180*s).
    lx0, lx1 = int(20 * s), int(180 * s)  # left plate x: 20..180
    rx0, rx1 = int(180 * s), int(340 * s)  # right plate x: 180..340 (rx0 == lx1)
    ly0, ly1 = int(20 * s), int(180 * s)  # shared y band: 20..180

    # left plate – F
    canvas[ly0:ly1, lx0:lx1] = 0.15
    canvas[ly0:ly1, lx0 : lx0 + bw] = 1.0
    canvas[ly0 : ly0 + bw, lx0 : lx1 - int(10 * s)] = 1.0
    canvas[ly0 + int(80 * s) : ly0 + int(80 * s) + bw, lx0 : lx0 + int(80 * s)] = 1.0

    # right plate – E
    canvas[ly0:ly1, rx0:rx1] = 0.15
    canvas[ly0:ly1, rx0 : rx0 + bw] = 1.0
    canvas[ly0 : ly0 + bw, rx0:rx1] = 1.0
    canvas[ly0 + int(80 * s) : ly0 + int(80 * s) + bw, rx0 : rx0 + int(100 * s)] = 1.0
    canvas[ly1 - bw : ly1, rx0:rx1] = 1.0

    pts6 = np.float32(
        [
            [lx0, ly0],  # p0 TL-left
            [rx0, ly0],  # p1 shared inner top  (= lx1)
            [rx1, ly0],  # p2 TR-right
            [rx1, ly1],  # p3 BR-right
            [rx0, ly1],  # p4 shared inner bottom (= lx1)
            [lx0, ly1],  # p5 BL-left
        ]
    )
    return canvas, pts6


def _distort(
    img: NDArray[np.float32],
    shift: int = 20,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Apply a mild perspective warp.

    Returns
    -------
    warped   : distorted image (same size as input)
    corners  : (4, 2) float32 plate corners [TL, TR, BR, BL] in warped-image coords
    """
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [shift, 0],
            [w - 1, shift // 2],
            [w - 1 - shift // 2, h - 1],
            [0, h - 1 - shift],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    corners = dst  # the destination IS the plate boundary after warping
    return warped, corners


def _distort_with_pts(
    img: NDArray[np.float32],
    pts: NDArray[np.float32],
    shift: int = 15,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Distort *img* and transform *pts* through the same homography."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [shift, 0],
            [w - 1, shift // 2],
            [w - 1 - shift // 2, h - 1],
            [0, h - 1 - shift],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    pts_warped = cv2.perspectiveTransform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
    return warped, pts_warped.astype(np.float32)


def _forward_rotate_points(
    pts: NDArray[np.float32],
    original_hw: tuple[int, int],
    k: int,
) -> NDArray[np.float32]:
    """
    Map corner coordinates from raw-frame space into the k-times-CCW-rotated
    display space.  Inverse of rotate_points() — used to compute what corners
    look like to the user when the camera is mounted at angle k.

    Each 90° CCW step applies: (x, y) → (y, W−1−x), then H,W swap.
    """
    H, W = original_hw
    pts = pts.copy()
    for _ in range(k % 4):
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = y
        pts[:, 1] = W - 1 - x
        H, W = W, H
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Mock experiment – no GUI, no CSV files needed
# ─────────────────────────────────────────────────────────────────────────────


class _MockData:
    """Minimal DataClass-compatible stub backed by a numpy array."""

    def __init__(self, frame: NDArray[np.float32]) -> None:
        self._frame = frame.astype(np.float32)
        self.data_numbers = [0]

    def get_frame(self, framenr: int, rotation_index: int) -> NDArray[np.float32]:
        return np.rot90(self._frame, k=rotation_index)

    def get_raw_frame(self, framenr: int) -> NDArray[np.float32]:
        return self._frame

    def get_frame_count(self) -> int:
        return 1


class _MockExperiment:
    """Minimal Experiment stub that dewarping functions can talk to."""

    def __init__(self, frame: NDArray[np.float32], tmp_path: str) -> None:
        self._data = _MockData(frame)
        self.folder_path = tmp_path
        self.exp_name = "mock"
        self._h5_file: Any = None
        self.h5_path: str | None = None

    def get_data(self, datatype: str) -> _MockData:  # noqa: ARG002
        return self._data

    @property
    def h5_file(self) -> Any:
        return self._h5_file

    @h5_file.setter
    def h5_file(self, value: Any) -> None:
        self._h5_file = value


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _rmse(a: NDArray, b: NDArray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    # Resize b to match a if needed (minor size differences from buffer factor)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _run_generator(gen) -> None:
    for _ in gen:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# LFS tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rotation_index", [0, 1, 2, 3])
def test_lfs_dewarping_rotation(tmp_path, rotation_index):
    """
    Dewarped LFS output must be close to the clean original for all 4 rotations.

    Pipeline:
      clean F  →  perspective distort  →  simulate camera rotation
        →  mark corners in display  →  dewarp_lateral_flame_spread
        →  compare dewarped vs clean F
    """
    clean = _make_f(200)
    distorted, corners_orig = _distort(clean, shift=10)

    # corners_orig are [TL, TR, BR, BL] in `distorted` image coords.
    # _distort() returns the destination corners which ARE the plate boundary
    # in the distorted image.  The GUI always counter-rotates the raw frame
    # back to upright, so what the user sees — and marks — is always `distorted`.
    # Corner coordinates are therefore always in `distorted` space regardless of
    # rotation_index.
    corners_orig_f32 = np.asarray(corners_orig, dtype=np.float32)

    # Simulate camera mounted at rotation_index * 90° CW.
    # The GUI applies the inverse CCW rotation so the display shows `distorted`.
    camera_raw = np.rot90(distorted, k=(4 - rotation_index) % 4)

    # Build mock experiment around the RAW (rotated) camera frame
    exp = _MockExperiment(camera_raw, str(tmp_path))

    target_h, target_w = clean.shape
    cfg = DewarpConfig(
        target_ratio=target_h / target_w,
        target_pixels_width=target_w,
        target_pixels_height=target_h,
        plate_width_mm=500.0,
        plate_height_mm=500.0,
        rotation_index=rotation_index,
        datatype="IR",
    )

    _run_generator(dewarp_lateral_flame_spread(exp, corners_orig_f32.tolist(), cfg))

    with h5py.File(exp.h5_path, "r") as f:
        dewarped = np.asarray(f["dewarped_data"]["data"][:, :, 0], dtype=np.float32)

    rmse = _rmse(clean, dewarped)
    assert (
        rmse < RMSE_THRESHOLD
    ), f"LFS rotation_index={rotation_index}: RMSE={rmse:.4f} >= {RMSE_THRESHOLD}"


# ─────────────────────────────────────────────────────────────────────────────
# Room Corner tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rotation_index", [0, 1, 2, 3])
def test_rce_dewarping_rotation(tmp_path, rotation_index):
    """
    Dewarped RCE left (F) and right (E) plates must be close to their clean
    originals for all 4 rotation indices.

    Pipeline:
      clean F+E canvas  →  perspective distort  →  simulate camera rotation
        →  mark 6 corners in display  →  rotate_points  →  dewarp_room_corner_remap
        →  compare left/right dewarped vs clean plate crops
    """
    SIZE = 200
    canvas, pts6_orig = _make_fe(SIZE)
    distorted, pts6_warped = _distort_with_pts(canvas, pts6_orig, shift=15)

    # The raw frame stored in production is always the unrotated camera output.
    # Rotation is applied only in get_frame() for display; get_raw_frame() always
    # returns the unrotated frame.  The mock mirrors this exactly.
    camera_raw = distorted  # never pre-rotated

    # Build mock experiment.  get_frame(0, ri) returns np.rot90(camera_raw, ri),
    # which is the display the user sees when the camera is mounted at ri * 90° CW.
    exp = _MockExperiment(camera_raw, str(tmp_path))

    # The user marks the 6 corners in the DISPLAY coordinate system.
    # _forward_rotate_points converts raw-frame corners → display corners.
    pts6_display = _forward_rotate_points(
        pts6_warped, distorted.shape[:2], rotation_index
    )

    s = SIZE / 200
    plate_h = int(160 * s)
    plate_w = int(160 * s)

    cfg = DewarpConfig(
        target_ratio=plate_h / plate_w,
        target_pixels_width=plate_w,
        target_pixels_height=plate_h,
        plate_width_mm=500.0,
        plate_height_mm=500.0,
        rotation_index=rotation_index,
        datatype="IR",
    )

    _run_generator(dewarp_room_corner_remap(exp, pts6_display.tolist(), cfg))

    # Reference: undistorted clean plate crops (must match _make_fe layout)
    s = SIZE / 200
    lx0, lx1 = int(20 * s), int(180 * s)  # left plate x: 20..180
    rx0, rx1 = int(180 * s), int(340 * s)  # right plate x: 180..340
    ly0, ly1 = int(20 * s), int(180 * s)  # shared y band

    clean_left = canvas[ly0:ly1, lx0:lx1]
    clean_right = canvas[ly0:ly1, rx0:rx1]

    with h5py.File(exp.h5_path, "r") as f:
        dewarped_left = np.asarray(
            f["dewarped_data_left"]["data"][:, :, 0], dtype=np.float32
        )
        dewarped_right = np.asarray(
            f["dewarped_data_right"]["data"][:, :, 0], dtype=np.float32
        )

    rmse_l = _rmse(clean_left, dewarped_left)
    rmse_r = _rmse(clean_right, dewarped_right)

    assert (
        rmse_l < RMSE_THRESHOLD
    ), f"RCE left  rotation_index={rotation_index}: RMSE={rmse_l:.4f} >= {RMSE_THRESHOLD}"
    assert (
        rmse_r < RMSE_THRESHOLD
    ), f"RCE right rotation_index={rotation_index}: RMSE={rmse_r:.4f} >= {RMSE_THRESHOLD}"
