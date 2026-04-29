"""
Visual test for dewarping correctness across 4 camera orientations,
including the counter-rotation (Gegenrotation) applied via the GUI dropdown.

Scenario
--------
The scene is always the same: letter F with mild perspective distortion.
The camera is simulated as being mounted at 4 different angles
(0°, 90° CW, 180°, 270° CW), which makes the raw frame appear rotated.

The user corrects this via the rotation_index dropdown (0 / 1 / 2 / 3),
which applies a matching CCW rotation via get_frame(idx, rotation_index).
After correction all 4 frames look identical.
The user then marks the same plate corners on the corrected frame,
and the dewarping should produce the same undistorted result for all 4 cases.

Rows
----
1. Raw camera frame  (different orientation per column)
2. After Gegenrotation  (should look identical in all 4 columns)
3. Dewarped output  (should look identical in all 4 columns)
4. Reference: clean unwarped F

Run from the repo root:
    python tests/integration/test_dewarping_rotation_visual.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from flametrack.analysis.ir_analysis import (
    compute_remap_from_homography,
    get_dewarp_parameters,
)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def create_f_image(size: int = 260) -> np.ndarray:
    """Draw the letter F as a float32 image."""
    img = np.zeros((size, size), dtype=np.float32)
    s = size / 260
    bar_w = int(40 * s)
    img[int(40 * s) : int(220 * s), int(30 * s) : int(30 * s) + bar_w] = 1.0  # vertical
    img[int(40 * s) : int(40 * s) + bar_w, int(30 * s) : int(200 * s)] = 1.0  # top bar
    img[int(130 * s) : int(130 * s) + bar_w, int(30 * s) : int(165 * s)] = (
        1.0  # mid bar
    )
    return img


def apply_perspective_distortion(
    img: np.ndarray, shift: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp the image with a mild perspective transform to simulate a
    slightly tilted camera.

    Returns
    -------
    warped   – distorted image (same canvas size as input)
    corners  – (4,2) float32: plate corners [TL, TR, BR, BL] in warped image
    """
    h, w = img.shape[:2]
    corners_dst = np.float32(
        [
            [shift, 0],  # TL
            [w - 1, shift // 2],  # TR
            [w - 1 - shift // 2, h - 1],  # BR
            [0, h - 1 - shift],  # BL
        ]
    )
    corners_src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(corners_src, corners_dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped, corners_dst


# ---------------------------------------------------------------------------
# Camera simulation
# ---------------------------------------------------------------------------


def simulate_camera_mounted_at(img: np.ndarray, mount_cw_steps: int) -> np.ndarray:
    """
    Simulate a camera mounted rotated by mount_cw_steps * 90° clockwise.
    CW rotation by k steps  =  CCW rotation by (4-k) steps.
    """
    return np.rot90(img, k=(4 - mount_cw_steps) % 4)


def apply_counter_rotation(raw_frame: np.ndarray, rotation_index: int) -> np.ndarray:
    """
    Mimic get_frame(idx, rotation_index): rotate the raw frame by
    rotation_index * 90° CCW to compensate the camera tilt.
    """
    return np.rot90(raw_frame, k=rotation_index)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def sort_corners_tl_tr_br_bl(corners: np.ndarray) -> np.ndarray:
    """Return the 4 corners in clockwise order: TL, TR, BR, BL."""
    cx, cy = corners.mean(axis=0)
    ordered: list = [None, None, None, None]
    for pt in corners:
        if pt[0] <= cx and pt[1] <= cy:
            ordered[0] = pt
        elif pt[0] > cx and pt[1] <= cy:
            ordered[1] = pt
        elif pt[0] > cx and pt[1] > cy:
            ordered[2] = pt
        else:
            ordered[3] = pt
    if any(p is None for p in ordered):
        angles = np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx)
        return corners[np.argsort(angles)]
    return np.array(ordered, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dewarping
# ---------------------------------------------------------------------------


def dewarp(
    img: np.ndarray,
    corners: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """
    Dewarp *img* so that the quadrilateral defined by *corners*
    [TL, TR, BR, BL] maps to a [target_w × target_h] rectangle.
    """
    sorted_corners = sort_corners_tl_tr_br_bl(corners)
    params = get_dewarp_parameters(
        sorted_corners,
        target_pixels_width=target_w,
        target_pixels_height=target_h,
    )
    H_fwd = params["transformation_matrix"]
    H_inv = np.linalg.inv(H_fwd).astype(np.float32)
    src_x, src_y = compute_remap_from_homography(H_inv, target_w, target_h)
    src_x = np.clip(src_x, 0, img.shape[1] - 1)
    src_y = np.clip(src_y, 0, img.shape[0] - 1)
    map_x, map_y = cv2.convertMaps(src_x, src_y, cv2.CV_16SC2)
    return cv2.remap(img.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    SIZE = 300  # canvas size
    TARGET = 260  # dewarped output size

    f_clean = create_f_image(TARGET)
    f_warped, plate_corners = apply_perspective_distortion(
        create_f_image(SIZE), shift=30
    )

    # The corners are always the same in the corrected (Gegenrotation) frame.
    # The corrected frame always looks like f_warped (same orientation).

    LABELS = [
        "0° (no mount rotation)",
        "90° CW mount  →  rotation_index=1",
        "180° mount  →  rotation_index=2",
        "270° CW mount  →  rotation_index=3",
    ]

    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.suptitle(
        "FlameTrack Dewarping Test – Camera Rotation + Gegenrotation\n"
        "Row 1: raw camera image (tilted mount)  "
        "│  Row 2: after Gegenrotation (dropdown)  "
        "│  Row 3: dewarped  │  Row 4: reference",
        fontsize=12,
        fontweight="bold",
    )

    for col, (mount_cw, label) in enumerate(zip(range(4), LABELS)):
        rotation_index = mount_cw  # user sets this in the dropdown

        # ── Row 0: raw frame (camera mounted tilted) ──────────────────
        raw = simulate_camera_mounted_at(f_warped, mount_cw_steps=mount_cw)

        ax = axes[0, col]
        ax.imshow(raw, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Raw – {label}", fontsize=8, wrap=True)
        ax.axis("off")

        # ── Row 1: after Gegenrotation (= corrected frame) ────────────
        corrected = apply_counter_rotation(raw, rotation_index)
        # corners are always in corrected-frame coordinates = plate_corners

        ax = axes[1, col]
        ax.imshow(corrected, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"After Gegenrotation (idx={rotation_index})", fontsize=9)
        sorted_c = sort_corners_tl_tr_br_bl(plate_corners)
        poly = patches.Polygon(
            sorted_c, closed=True, edgecolor="red", facecolor="none", lw=1.5
        )
        ax.add_patch(poly)
        for i, (cx, cy) in enumerate(sorted_c):
            ax.plot(cx, cy, "r+", markersize=9, markeredgewidth=2)
            ax.annotate(
                ["TL", "TR", "BR", "BL"][i],
                (cx, cy),
                color="red",
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.axis("off")

        # ── Row 2: dewarped ───────────────────────────────────────────
        f_out = dewarp(corrected, plate_corners, target_w=TARGET, target_h=TARGET)
        rmse = float(np.sqrt(np.mean((f_out - f_clean) ** 2)))

        ax = axes[2, col]
        ax.imshow(f_out, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Dewarped  (RMSE={rmse:.4f})", fontsize=9)
        ax.axis("off")

        # ── Row 3: reference ──────────────────────────────────────────
        ax = axes[3, col]
        ax.imshow(f_clean, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Reference (clean F)", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "_outputs" / "dewarping_rotation_test.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
