"""
Visual test for Room Corner dewarping correctness across all 4 rotation indices.

Tests the exact code path used in production:
  - rotate_points()  to convert GUI corners → original image coords
  - get_dewarp_parameters() + _compute_remap_maps()
  - cv2.remap() on the raw (unrotated) frame

For each rotation_index the dewarped left and right plates should look
identical, regardless of how the "camera" was mounted.

Run from the repo root:
    python tests/integration/test_roomcorner_rotation_visual.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from flametrack.analysis.ir_analysis import (
    compute_remap_from_homography,
    get_dewarp_parameters,
)
from flametrack.gui.plotting_utils import rotate_points

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_remap_maps(homography, w, h):
    """Invert homography and compute remap grids (mirrors production code)."""
    H_inv = np.linalg.inv(homography).astype(np.float32)
    src_x, src_y = compute_remap_from_homography(H_inv, w, h)
    return src_x.astype(np.float32), src_y.astype(np.float32)


def dewarp_plate(raw_frame, corners_original, target_w, target_h):
    """Dewarp one plate given corners in original (unrotated) image coords."""
    params = get_dewarp_parameters(
        corners_original,
        target_pixels_width=target_w,
        target_pixels_height=target_h,
    )
    src_x, src_y = _compute_remap_maps(
        params["transformation_matrix"], target_w, target_h
    )
    h_in, w_in = raw_frame.shape[:2]
    map_x, map_y = cv2.convertMaps(
        np.clip(src_x, 0, w_in - 1),
        np.clip(src_y, 0, h_in - 1),
        cv2.CV_16SC2,
    )
    return cv2.remap(raw_frame.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)


def forward_rotate_points(corners, original_hw, k):
    """
    Forward rotation: original image coords → rotated display coords.
    Inverse of rotate_points().
    """
    H, W = original_hw
    pts = np.array(corners, dtype=np.float32)
    for _ in range(k % 4):
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = y
        pts[:, 1] = W - 1 - x
        H, W = W, H
    return pts


# ---------------------------------------------------------------------------
# Synthetic scene
# ---------------------------------------------------------------------------


def create_room_corner_scene(canvas_h=300, canvas_w=400):
    """
    Two adjacent plates (left = letter F, right = letter E) on a canvas,
    with slight perspective distortion applied to the whole canvas.

    Returns
    -------
    raw_frame  : the (possibly warped) canvas as float32 image
    pts6       : 6 corner points [p0..p5] in canvas (unrotated) coordinates
    plate_hw   : (plate_h, plate_w) target output size
    """
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    s = canvas_h / 300

    # ── Left plate: letter F ──────────────────────────────────────────────
    # Plate occupies columns 50..170, rows 70..230
    lx0, lx1, ly0, ly1 = int(50 * s), int(170 * s), int(70 * s), int(230 * s)
    canvas[ly0:ly1, lx0:lx1] = 0.15  # plate background
    bw = int(15 * s)  # stroke width
    canvas[ly0:ly1, lx0 : lx0 + bw] = 0.9  # vertical bar
    canvas[ly0 : ly0 + bw, lx0 : lx1 - int(10 * s)] = 0.9  # top bar
    canvas[ly0 + int(65 * s) : ly0 + int(65 * s) + bw, lx0 : lx0 + int(50 * s)] = (
        0.9  # mid bar
    )

    # ── Right plate: letter E ─────────────────────────────────────────────
    rx0, rx1, ry0, ry1 = int(230 * s), int(350 * s), int(70 * s), int(230 * s)
    canvas[ry0:ry1, rx0:rx1] = 0.15
    canvas[ry0:ry1, rx0 : rx0 + bw] = 0.9  # vertical bar
    canvas[ry0 : ry0 + bw, rx0:rx1] = 0.9  # top bar
    canvas[ry0 + int(65 * s) : ry0 + int(65 * s) + bw, rx0 : rx0 + int(80 * s)] = (
        0.9  # mid bar
    )
    canvas[ry1 - bw : ry1, rx0:rx1] = 0.9  # bottom bar

    # ── 6-point Room Corner layout ────────────────────────────────────────
    # p0---p1---p2
    # |  L |  R |
    # p5---p4---p3
    pts6 = np.float32(
        [
            [lx0, ly0],  # p0: TL of left plate
            [rx0, ly0],  # p1: shared inner top  (= TR-left = TL-right)
            [rx1, ry0],  # p2: TR of right plate
            [rx1, ry1],  # p3: BR of right plate
            [rx0, ly1],  # p4: shared inner bottom
            [lx0, ly1],  # p5: BL of left plate
        ]
    )

    plate_h = int((ly1 - ly0) * 0.9)
    plate_w = int((lx1 - lx0) * 0.9)

    # ── Slight perspective distortion ─────────────────────────────────────
    d = int(15 * s)
    src = np.float32(
        [[0, 0], [canvas_w - 1, 0], [canvas_w - 1, canvas_h - 1], [0, canvas_h - 1]]
    )
    dst = np.float32(
        [
            [d, 0],
            [canvas_w - 1, d // 2],
            [canvas_w - 1 - d // 2, canvas_h - 1],
            [0, canvas_h - 1 - d],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    raw_frame = cv2.warpPerspective(canvas, M, (canvas_w, canvas_h))
    pts6_warped = cv2.perspectiveTransform(pts6.reshape(1, -1, 2), M).reshape(-1, 2)

    return raw_frame, pts6_warped.astype(np.float32), (plate_h, plate_w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    raw_frame, pts6_orig, (plate_h, plate_w) = create_room_corner_scene()
    orig_hw = raw_frame.shape[:2]  # (H, W) of unrotated frame

    LABELS = [
        "rotation_index=0\n(no mount rotation)",
        "rotation_index=1\n(camera 90° CW)",
        "rotation_index=2\n(camera 180°)",
        "rotation_index=3\n(camera 270° CW)",
    ]

    # 4 rows: raw | corrected display | dewarped-left | dewarped-right
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.suptitle(
        "Room Corner Dewarping – rotate_points() Rotation Test\n"
        "Each column simulates a different camera mount angle.\n"
        "Rows: raw frame | corrected display (what user sees) | "
        "dewarped left plate | dewarped right plate",
        fontsize=11,
        fontweight="bold",
    )

    for col, (k, label) in enumerate(zip(range(4), LABELS)):
        # ── Simulate camera mounted at k*90° CW ─────────────────────────
        # Raw frame appears rotated (4-k)*90° CCW = k*90° CW from user view
        camera_raw = np.rot90(raw_frame, k=(4 - k) % 4)

        # ── Row 0: raw camera frame ─────────────────────────────────────
        axes[0, col].imshow(camera_raw, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Raw camera\n{label}", fontsize=8)
        axes[0, col].axis("off")

        # ── Corrected display: apply rotation_index (= k CCW) ───────────
        corrected = np.rot90(camera_raw, k=k)
        # corrected should look identical to raw_frame for all k

        # Forward-rotate the original corners to corrected display coords
        # (this is what the user sees and clicks on)
        pts6_in_corrected = forward_rotate_points(pts6_orig, orig_hw, k)

        # ── Row 1: corrected display with marked corners ─────────────────
        ax = axes[1, col]
        ax.imshow(corrected, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Corrected display\n(after rotation_index)", fontsize=8)
        ax.plot(
            pts6_in_corrected[:, 0],
            pts6_in_corrected[:, 1],
            "r+",
            markersize=10,
            markeredgewidth=2,
        )
        for i, (px, py) in enumerate(pts6_in_corrected):
            ax.annotate(
                f"p{i}",
                (px, py),
                color="red",
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.axis("off")

        # ── Production code path: rotate_points → dewarp raw frame ───────
        # The corrected frame has shape orig_hw (same as unrotated raw_frame)
        # because rot90(rot90(raw, 4-k), k) = raw.
        # rotate_points expects the UNROTATED frame shape.
        pts_left_corrected = pts6_in_corrected[[0, 1, 4, 5]]
        pts_right_corrected = pts6_in_corrected[[1, 2, 3, 4]]

        # rotate_points: corrected-display coords → original raw-frame coords
        sel_left = np.array(
            rotate_points(pts_left_corrected, orig_hw, k), dtype=np.float32
        )
        sel_right = np.array(
            rotate_points(pts_right_corrected, orig_hw, k), dtype=np.float32
        )

        # Dewarp on the ORIGINAL raw_frame (not camera_raw, not corrected)
        # This is exactly what dewarp_room_corner_remap does:
        #   raw = data.get_raw_frame(idx)
        left_out = dewarp_plate(raw_frame, sel_left, plate_w, plate_h)
        right_out = dewarp_plate(raw_frame, sel_right, plate_w, plate_h)

        # ── Row 2: dewarped left plate ────────────────────────────────────
        axes[2, col].imshow(left_out, cmap="gray", vmin=0, vmax=1)
        axes[2, col].set_title("Dewarped – left plate (F)", fontsize=8)
        axes[2, col].axis("off")

        # ── Row 3: dewarped right plate ───────────────────────────────────
        axes[3, col].imshow(right_out, cmap="gray", vmin=0, vmax=1)
        axes[3, col].set_title("Dewarped – right plate (E)", fontsize=8)
        axes[3, col].axis("off")

    plt.tight_layout()
    out_path = (
        Path(__file__).parent.parent / "_outputs" / "roomcorner_rotation_test.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
