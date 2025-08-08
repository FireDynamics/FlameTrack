import numpy as np
from skimage.metrics import structural_similarity as ssim


def assert_image_similarity(
    img_a, img_b, *, ssim_thresh=0.95, mae_thresh=10.0, name=""
):
    """
    Vergleicht zwei Bilder anhand von SSIM und MAE.

    :param img_a: erstes Bild (z. B. Original)
    :param img_b: zweites Bild (z. B. Dewarped)
    :param ssim_thresh: Mindestwert für strukturelle Ähnlichkeit
    :param mae_thresh: maximaler mittlerer Fehler (MAE)
    :param name: optionale Bezeichnung für Debug-Ausgaben
    """
    img_a = img_a.astype(np.float32)
    img_b = img_b.astype(np.float32)

    # Dynamischer Wertebereich (z. B. 255.0 bei 8-Bit-Bildern)
    data_range = float(np.max(img_a) - np.min(img_a))
    if data_range == 0:
        data_range = 1.0  # Sicherheitsfallback bei konstantem Bild

    # SSIM
    score = ssim(img_a, img_b, data_range=data_range)
    assert (
        score >= ssim_thresh
    ), f"SSIM für {name} zu gering: {score:.4f} < {ssim_thresh}"

    # MAE
    mae = np.mean(np.abs(img_a - img_b))
    assert mae <= mae_thresh, f"MAE für {name} zu hoch: {mae:.2f} > {mae_thresh}"
