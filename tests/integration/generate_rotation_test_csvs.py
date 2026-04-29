"""
Generates a single synthetic IR CSV file with 4 quadrants for rotation testing.

Layout (each quadrant = 300×300 px):
    ┌─────────────┬─────────────┐
    │  TL: 0°     │  TR: 90° CW │
    │  (idx=0)    │  (idx=1)    │
    ├─────────────┼─────────────┤
    │  BL: 180°   │  BR: 270°CW │
    │  (idx=2)    │  (idx=3)    │
    └─────────────┴─────────────┘

Each quadrant shows the same scene (F = left plate, E = right plate) as seen
by a camera mounted at the indicated angle.

Usage:
    python tests/integration/generate_rotation_test_csvs.py

Output (in tests/_outputs/rotation_test_data/):
    rotation_test_quadrants.csv   – 600×600 px image with 4 quadrants

How to test in FlameTrack:
    1. Open rotation_test_quadrants.csv as an IR dataset.
    2. Leave rotation_index = 0 (image is already upright as a whole).
    3. Mark 6 Room Corner points on one quadrant at a time.
       Use the coordinates printed by this script.
    4. For the TL quadrant (0°) you need no mental correction.
       For TR (90° CW) the scene is rotated – mark the letters as they
       appear but expect the same dewarped output as TL.
       Alternatively: set rotation_index to the matching value and mark
       the corners on the corrected (upright) view using the same coords.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------

T_BACKGROUND = 20.0  # °C – cold background
T_PLATE_BG = 35.0  # °C – warm plate surface
T_STROKE = 80.0  # °C – hot letter strokes

QUAD_SIZE = 300  # each quadrant is QUAD_SIZE × QUAD_SIZE pixels


def create_scene(size: int = QUAD_SIZE) -> np.ndarray:
    """
    Create a synthetic IR temperature image (float32, °C) showing:
      - Left plate:  letter F  (~25–57% of width, ~23–77% of height)
      - Right plate: letter E  (~58–87% of width, ~23–77% of height)
    with a slight perspective distortion applied.
    """
    s = size / 300
    canvas = np.full((size, size), T_BACKGROUND, dtype=np.float32)
    bw = max(1, int(15 * s))  # stroke width

    # ── Left plate (F) ───────────────────────────────────────────────────
    lx0, lx1 = int(50 * s), int(140 * s)
    ly0, ly1 = int(70 * s), int(230 * s)
    canvas[ly0:ly1, lx0:lx1] = T_PLATE_BG
    canvas[ly0:ly1, lx0 : lx0 + bw] = T_STROKE  # vertical
    canvas[ly0 : ly0 + bw, lx0 : lx1 - int(10 * s)] = T_STROKE  # top bar
    canvas[ly0 + int(65 * s) : ly0 + int(65 * s) + bw, lx0 : lx0 + int(50 * s)] = (
        T_STROKE  # mid bar
    )

    # ── Right plate (E) ───────────────────────────────────────────────────
    rx0, rx1 = int(160 * s), int(250 * s)
    ry0, ry1 = int(70 * s), int(230 * s)
    canvas[ry0:ry1, rx0:rx1] = T_PLATE_BG
    canvas[ry0:ry1, rx0 : rx0 + bw] = T_STROKE  # vertical
    canvas[ry0 : ry0 + bw, rx0:rx1] = T_STROKE  # top bar
    canvas[ry0 + int(65 * s) : ry0 + int(65 * s) + bw, rx0 : rx0 + int(80 * s)] = (
        T_STROKE  # mid bar
    )
    canvas[ry1 - bw : ry1, rx0:rx1] = T_STROKE  # bottom bar

    # ── Slight perspective distortion ─────────────────────────────────────
    d = int(15 * s)
    src = np.float32(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ]
    )
    dst = np.float32(
        [
            [d, 0],
            [size - 1, d // 2],
            [size - 1 - d // 2, size - 1],
            [0, size - 1 - d],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(canvas, M, (size, size))

    return warped


def get_corner_points(size: int = QUAD_SIZE):
    """
    Return the 6 Room Corner reference points in the UPRIGHT (0°) scene.

    Layout:
        p0 ---p1--- p2
        |   F  |  E  |
        p5 ---p4--- p3
    """
    s = size / 300
    lx0 = int(50 * s)
    lx1 = int(140 * s)
    rx0 = int(160 * s)
    rx1 = int(250 * s)
    ly0 = int(70 * s)
    ly1 = int(230 * s)
    return [
        (lx0, ly0),  # p0: TL of left plate
        (rx0, ly0),  # p1: shared inner top
        (rx1, ly0),  # p2: TR of right plate
        (rx1, ly1),  # p3: BR of right plate
        (rx0, ly1),  # p4: shared inner bottom
        (lx0, ly1),  # p5: BL of left plate
    ]


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def array_to_ir_csv(img: np.ndarray, path: Path) -> None:
    """
    Write a float32 temperature array as an IR CSV in FlameTrack format.
    Decimal separator: comma.  Field separator: semicolon.  Encoding: latin-1.
    """
    h, w = img.shape
    path.parent.mkdir(parents=True, exist_ok=True)

    t_min = float(img.min())
    t_max = float(img.max())

    with open(path, "w", encoding="latin-1", newline="\n") as f:
        f.write("[Settings]\n")
        f.write("Version=3\n")
        f.write(f"ImageWidth={w}\n")
        f.write(f"ImageHeight={h}\n")
        f.write(f"ShotRange={t_min:.2f};{t_max:.2f}\n".replace(".", ","))
        f.write("CalibRange=0,00;500,00\n")
        f.write("TempUnit=C\n")
        f.write("\n")
        f.write("[MeasDefs]\n")
        f.write("ID=M-Wert;Min;Max;Bereich;St.Abw.\n")
        f.write("\n")
        f.write("[Data]\n")

        for row in img:
            line = ";".join(f"{v:.2f}".replace(".", ",") for v in row)
            f.write(line + ";\n")

    print(f"  Written: {path}  ({w}×{h} px)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).parent.parent / "_outputs" / "rotation_test_data"
    scene = create_scene(QUAD_SIZE)
    Q = QUAD_SIZE

    # Build 2×2 grid: [row, col] → rotation_index (= CW steps)
    # TL=0°  TR=90°CW  BL=180°  BR=270°CW
    grid = np.full((2 * Q, 2 * Q), T_BACKGROUND, dtype=np.float32)

    configs = [
        (0, (0, 0), "0°    (rotation_index=0)  → top-left quadrant"),
        (1, (0, Q), "90°CW (rotation_index=1)  → top-right quadrant"),
        (2, (Q, 0), "180°  (rotation_index=2)  → bottom-left quadrant"),
        (3, (Q, Q), "270°CW(rotation_index=3)  → bottom-right quadrant"),
    ]

    for cw_steps, (row_off, col_off), desc in configs:
        # np.rot90 with k rotates CCW; CW by n steps = CCW by (4-n) steps
        rotated = np.rot90(scene, k=(4 - cw_steps) % 4)
        rh, rw = rotated.shape
        grid[row_off : row_off + rh, col_off : col_off + rw] = rotated

    # ── Central up-arrow, centred on the whole 600×600 grid ───────────────
    # Drawn AFTER compositing so it spans all four quadrants as one symbol.
    # It always points up and is the same in every load – it is a global
    # "north" marker, not part of any individual quadrant scene.
    cx = Q  # horizontal centre = 300
    cy = Q  # vertical centre   = 300
    shaft_hw = 8  # half-width of shaft in px
    shaft_len = 120  # total length of shaft
    head_hw = 30  # half-width of arrowhead base
    head_len = 55  # height of arrowhead

    shaft_top = cy - shaft_len // 2
    shaft_bot = cy + shaft_len // 2
    tip_row = shaft_top - head_len

    # Shaft
    grid[shaft_top:shaft_bot, cx - shaft_hw : cx + shaft_hw + 1] = T_STROKE

    # Arrowhead (filled triangle pointing up)
    for row in range(tip_row, shaft_top + 1):
        frac = (row - tip_row) / head_len
        hw = int(frac * head_hw)
        grid[row, cx - hw : cx + hw + 1] = T_STROKE

    out_path = out_dir / "rotation_test_quadrants.csv"
    print(f"Output directory: {out_dir}\n")
    array_to_ir_csv(grid, out_path)

    # ── Print reference corner coordinates ────────────────────────────────
    pts_upright = get_corner_points(QUAD_SIZE)

    print()
    print("=" * 66)
    print("Reference 6-point coordinates per quadrant")
    print("(mark these in the ROTATED quadrant as it appears in FlameTrack)")
    print("=" * 66)

    quadrant_labels = [
        ("Top-left    0°", 0, 0, 0),
        ("Top-right  90°CW", 0, Q, 1),
        ("Bot-left  180°", Q, 0, 2),
        ("Bot-right 270°CW", Q, Q, 3),
    ]

    for label, row_off, col_off, cw_steps in quadrant_labels:
        print(f"\n  {label}  (rotation_index={cw_steps})")
        print(f"  Quadrant origin in CSV: col={col_off}, row={row_off}")

        # Forward-rotate the upright points to match the rotated quadrant
        pts = np.float32(pts_upright)
        H, W = QUAD_SIZE, QUAD_SIZE
        for _ in range(cw_steps % 4):
            x, y = pts[:, 0].copy(), pts[:, 1].copy()
            # 90° CW: (x,y) → (H-1-y, x)  … then H,W swap
            pts[:, 0] = H - 1 - y
            pts[:, 1] = x
            H, W = W, H

        # Add quadrant offset
        pts_in_grid = pts + np.float32([col_off, row_off])

        names = [
            "p0(TL-L)",
            "p1(inner-top)",
            "p2(TR-R)",
            "p3(BR-R)",
            "p4(inner-bot)",
            "p5(BL-L)",
        ]
        for name, (px, py) in zip(names, pts_in_grid):
            print(f"    {name}: ({int(px):4d}, {int(py):4d})")

    print()
    print("TIP: To test rotation correction, open the CSV in FlameTrack,")
    print("     set rotation_index to match the quadrant you want to test,")
    print("     then mark the corner points on the corrected (upright) view")
    print("     using the upright coordinates:")
    print()
    print("  Upright reference points (same for all quadrants after correction):")
    for i, (px, py) in enumerate(pts_upright):
        names = [
            "p0(TL-L)",
            "p1(inner-top)",
            "p2(TR-R)",
            "p3(BR-R)",
            "p4(inner-bot)",
            "p5(BL-L)",
        ]
        print(f"    {names[i]}: ({px:4d}, {py:4d})")


if __name__ == "__main__":
    main()
