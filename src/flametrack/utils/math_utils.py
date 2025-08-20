import numpy as np


def compute_target_ratio(width: float, height: float) -> float:
    return height / width if width != 0 else 1.0


def estimate_resolution_from_points(
    p0,
    p1,
    p3,
    plate_width_mm,
    plate_height_mm,
    *,
    assumed_error_px: float = 0.5,
) -> dict:
    """
    Estimate spatial resolution and absolute measurement uncertainty based on manually marked rectangle.

    Parameters
    ----------
    p0 : array-like
        Top-left point of rectangle in pixels.
    p1 : array-like
        Top-right point of rectangle in pixels.
    p3 : array-like
        Bottom-left point of rectangle in pixels.
    plate_width_mm : float
        Real-world width of the physical plate in millimeters.
    plate_height_mm : float
        Real-world height of the physical plate in millimeters.
    assumed_error_px : float, optional
        Pixel accuracy assumed for manual point selection (default is ±0.5 px).

    Returns
    -------
    dict
        Dictionary with estimated resolution and error terms (all float):
        - mm_per_px_width
        - mm_per_px_height
        - error_mm_width
        - error_mm_height
        - assumed_pixel_error
    """

    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    p3 = np.asarray(p3)

    width_px = np.linalg.norm(p1 - p0)
    height_px = np.linalg.norm(p3 - p0)

    if width_px == 0 or height_px == 0:
        raise ValueError("Degenerate rectangle: width or height is zero.")

    mm_per_px_w = plate_width_mm / width_px
    mm_per_px_h = plate_height_mm / height_px

    error_mm_w = mm_per_px_w * assumed_error_px
    error_mm_h = mm_per_px_h * assumed_error_px

    return {
        "mm_per_px_width": mm_per_px_w,
        "mm_per_px_height": mm_per_px_h,
        "error_mm_width": error_mm_w,
        "error_mm_height": error_mm_h,
        "assumed_pixel_error": assumed_error_px,
    }
