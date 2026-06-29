Physics Background
==================

This document gives Mouni the physical foundation needed for both phases of
the internship: (1) correcting infrared temperature readings for material
emissivity, and (2) detecting the flame edge in IR/optical video. It assumes
a strong software background but limited physics, so every formula is
explained symbol-by-symbol and tied back to a concrete NumPy snippet.

.. contents::
   :local:
   :depth: 2


Part 1: Infrared Thermography and Emissivity Correction
--------------------------------------------------------

1.1 Why IR cameras measure apparent, not true temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An infrared (IR) camera does not measure temperature directly. Every object
above absolute zero emits electromagnetic radiation; the camera's sensor
measures the **radiance** :math:`L` reaching it (power per unit area per unit
solid angle) and converts that number into a temperature using a built-in
model. That model assumes the object is a **blackbody**.

A blackbody is an idealised surface that absorbs all incoming radiation and
re-emits the theoretical maximum amount of radiation possible at a given
temperature. No real material is a perfect blackbody — every real surface
emits *less* radiance than a blackbody at the same temperature. The ratio of
the two is called **emissivity** :math:`\varepsilon`:

.. math::

   \varepsilon = \frac{L_{\text{real surface}}(T)}{L_{\text{blackbody}}(T)}

Emissivity is dimensionless, :math:`0 < \varepsilon \le 1`, and depends on
the material, its surface finish (oxidised, painted, polished), and
sometimes the viewing angle and wavelength band. Representative values
relevant to fire/lateral-flame-spread testing (compiled from
[pitarma2019wood]_, [fang2021building]_, and the two-colour
pyrometry study of burning composites in [schartel2022twocolor]_):

.. list-table::
   :header-rows: 1

   * - Material
     - Typical ε (8–14 µm)
   * - Wood (untreated, natural)
     - 0.90 – 0.95
   * - Gypsum board / plasterboard
     - 0.90 – 0.95
   * - Concrete / masonry
     - 0.90 – 0.95
   * - Steel, heavily oxidised/rusted
     - 0.60 – 0.85
   * - Steel, lightly oxidised
     - 0.20 – 0.60
   * - Steel, polished/bare
     - 0.05 – 0.20
   * - Glass-fibre phenolic composite
     - ≈ 0.90
   * - Carbon-fibre phenolic composite
     - ≈ 0.86

The spread for steel is the practical headache: [schartel2022twocolor]_
measured oxidised steel plate emissivity ranging from 0.07 to 0.64 *on the
same specimen* depending on oxide layer thickness, which is why a single
assumed ε for "the rig" is not good enough (see :ref:`region-based-correction`).

If the camera assumes :math:`\varepsilon = 1` (blackbody) but the real
surface has, say, :math:`\varepsilon = 0.9`, the camera under-reads the
radiance and therefore under-reads the temperature. The displayed value is
the **apparent temperature** :math:`T_{\text{app}}`; the physical surface
temperature is the **true temperature** :math:`T_{\text{true}}`.

1.2 Planck's Law
~~~~~~~~~~~~~~~~~

The radiance emitted by a blackbody as a function of wavelength and
temperature is given by **Planck's Law** [planck1901]_:

.. math::

   B(\lambda, T) = \frac{2hc^2}{\lambda^5}
                   \cdot \frac{1}{e^{\,hc / (\lambda k_B T)} - 1}

where:

- :math:`\lambda` — wavelength [m]
- :math:`T` — absolute temperature [K]
- :math:`h = 6.626 \times 10^{-34}\ \text{J·s}` — Planck constant
- :math:`c = 3 \times 10^{8}\ \text{m/s}` — speed of light
- :math:`k_B = 1.381 \times 10^{-23}\ \text{J/K}` — Boltzmann constant
- :math:`B(\lambda, T)` — spectral radiance: power per unit area, per unit
  solid angle, per unit wavelength interval

A real surface emits :math:`L(\lambda, T) = \varepsilon(\lambda) \cdot
B(\lambda, T)`. Emissivity can vary with wavelength (*spectral emissivity*);
in this project we treat it as a single wavelength-averaged constant per
material region, which is the standard simplification for broadband
microbolometer cameras (e.g. the 7.5–14 µm long-wave-IR band typical of FLIR
research cameras).

A real IR camera does not see one wavelength — it integrates the incoming
spectral radiance over its detector's spectral response band
:math:`[\lambda_1, \lambda_2]`:

.. math::

   L_{\text{camera}}(T) = \int_{\lambda_1}^{\lambda_2} \varepsilon(\lambda)\, B(\lambda, T)\, d\lambda

Integrating Planck's law over *all* wavelengths (:math:`\lambda_1 \to 0`,
:math:`\lambda_2 \to \infty`) yields the **Stefan–Boltzmann Law**
([incropera2007]_, Ch. 12):

.. math::

   M = \varepsilon \cdot \sigma \cdot T^4

where :math:`M` is total radiated power per unit area and
:math:`\sigma = 5.670 \times 10^{-8}\ \text{W m}^{-2}\text{K}^{-4}` is the
Stefan–Boltzmann constant. This :math:`T^4` dependence is the reason
emissivity errors matter *much* more at fire temperatures (500–1300 K) than
at room temperature: a fixed percentage error in ε produces a temperature
error that grows with absolute temperature, and a fixed temperature error
near a flame front directly corrupts the detected edge location (Part 2).

1.3 The correction formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the Stefan–Boltzmann form, the camera's apparent temperature is the
temperature a *blackbody* would need in order to radiate the same power the
sensor actually measured from the real (non-blackbody) surface:

.. math::

   \sigma \cdot T_{\text{app}}^4 = \varepsilon \cdot \sigma \cdot T_{\text{true}}^4

Solving for the true temperature:

.. math::

   T_{\text{true}} = \frac{T_{\text{app}}}{\varepsilon^{1/4}}

This is the standard simplified correction used by camera vendors and
emissivity calculators (e.g. [calcacademy]_). Both temperatures must be
in **Kelvin** — the fourth-power relationship only holds on an absolute
scale.

**Caveat — reflected radiation.** In a lab fire-test environment the
surface also reflects ambient/background radiation (hot walls, other
burning surfaces, lighting), and the atmosphere between camera and target
absorbs/re-emits a small fraction of the signal. The fuller radiometric
model used in calibrated IR thermometry is:

.. math::

   L_{\text{measured}} = \varepsilon \cdot \tau \cdot B(T_{\text{true}})
                        + (1-\varepsilon) \cdot \tau \cdot L_{\text{refl}}
                        + (1-\tau) \cdot B(T_{\text{atm}})

where :math:`\tau` is the atmospheric transmittance (≈ 1 over the short
distances typical of lab rigs, so this term is usually dropped) and
:math:`L_{\text{refl}}` is the radiance of reflected background sources,
often approximated as blackbody radiation at an assumed "reflected
temperature" :math:`T_{\text{refl}}` (ambient/room temperature, unless a hot
wall or adjacent burner dominates the reflection). For most FlameTrack rigs
:math:`\tau \approx 1` and :math:`T_{\text{refl}} \approx T_{\text{ambient}}`,
so the simplified :math:`T_{\text{true}} = T_{\text{app}} / \varepsilon^{1/4}`
is adequate; the reflected term becomes important only for low-ε (shiny
metal) regions, where re-emitting little of its own radiation makes the
surface act almost like a mirror for its surroundings.

**Worked example.** A surface truly at 500 K with :math:`\varepsilon = 0.85`:

.. math::

   T_{\text{app}} = 500 \cdot 0.85^{1/4} \approx 500 \cdot 0.961 \approx 480.6\ \text{K}

So the camera reads ≈480.6 K for a surface that is truly 500 K — a ~19 K
error, growing rapidly for lower ε or higher T.

**Python implementation:**

.. code-block:: python

   import numpy as np

   def apply_emissivity_correction(T_app_K: np.ndarray, eps: float) -> np.ndarray:
       """Recover true temperature [K] from apparent temperature [K].

       T_app_K : apparent (camera-reported) temperature in Kelvin
       eps     : emissivity of the region, 0 < eps <= 1
       """
       return T_app_K / np.power(eps, 0.25)

   # worked example
   T_app = 480.6
   T_true = apply_emissivity_correction(np.array([T_app]), eps=0.85)
   print(T_true)  # -> [500.02]

.. _region-based-correction:

1.4 Why region-based correction is necessary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A room-corner or lateral flame-spread test image contains several materials
at once, each with its own ε (see the table in 1.1): the burning specimen
(wood/composite, ε ≈ 0.85–0.95), the steel mounting frame (ε ≈ 0.1–0.8
depending on oxidation, [schartel2022twocolor]_), and concrete or
gypsum board surroundings (ε ≈ 0.9–0.95). Flames themselves are
*semi-transparent* in IR — their effective emissivity depends on soot
loading and flame thickness, and is not a fixed material property, which is
why FlameTrack treats flame pixels as "not a calibrated surface" rather than
trying to emissivity-correct them.

Applying one global ε to the whole frame is correct for at most one of
these regions and systematically biased for the rest — typically
under-estimating steel temperature (since its true ε is much lower than a
wood-like default) while being roughly correct for the specimen itself.
**Region-based correction** — assigning a per-ROI ε and applying the
:math:`T_{\text{true}} = T_{\text{app}}/\varepsilon^{1/4}` correction
independently per mask — removes this systematic bias.

This is the goal of **Phase 1** of the internship: extend FlameTrack's
dewarped-frame pipeline with a per-region emissivity correction step, before
the corrected frames are handed to edge detection (Part 2):

.. code-block:: python

   import numpy as np

   def correct_frame(frame_K: np.ndarray, region_masks: dict[str, np.ndarray],
                      region_eps: dict[str, float]) -> np.ndarray:
       """Apply per-region emissivity correction to one IR frame.

       frame_K      : 2D array, apparent temperature in Kelvin
       region_masks : name -> boolean mask (same shape as frame_K)
       region_eps   : name -> emissivity for that region
       """
       corrected = frame_K.copy()
       for name, mask in region_masks.items():
           eps = region_eps[name]
           corrected[mask] = frame_K[mask] / np.power(eps, 0.25)
       return corrected

Open questions worth flagging early: how to handle pixels belonging to
*overlapping* ROIs, where to persist the chosen ε values (alongside results
in the HDF5 file, for reproducibility), and how to validate the correction
against a reference surface of known temperature and emissivity.


Part 2: Flame Edge Detection
-----------------------------

2.1 Problem definition in FlameTrack context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FlameTrack defines the **flame edge** as the lateral position of the flame
front along each row of the dewarped image, tracked over time. Concretely,
for an image of height :math:`H` rows, the edge at time step :math:`t` is a
1D array ``x[y]`` for ``y in range(H)``, and stacking these over all frames
gives a 2D array ``edge[t, y]`` (x-position in pixels or mm, per row, per
frame). Plotting ``edge`` against time at a fixed row — or its outer
envelope — gives the **flame spread curve**, the primary scientific
quantity FlameTrack exists to extract: flame spread *rate* is simply
:math:`dx/dt` of this curve.

The detection problem, per frame, is: given a 2D image (IR temperature field
or visible-light intensity), find ``x(y)`` — the row-wise flame boundary.

2.2 Classical approaches (current FlameTrack)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FlameTrack currently implements several classical, row-by-row methods:

1. For each row :math:`y`, extract the 1D intensity/temperature profile
   :math:`I(x)`.
2. Apply a threshold or gradient criterion to find the transition from
   "background" to "flame".
3. Optionally restrict the search using **Otsu's method**
   [otsu1979]_: compute a global threshold that splits the (assumed
   bimodal) intensity histogram into two classes, then search only within
   the resulting flame mask.

Concrete variants in use: a fixed threshold (edge at the first :math:`x`
where :math:`I(x) > T_{\text{fixed}}`), a gradient/derivative criterion
(edge at the :math:`x` of steepest :math:`dI/dx`), and Otsu-masked
thresholding. These mirror the classical color/intensity/gradient-based
flame detectors surveyed for visible-light fire imaging
[wildfireedge2025]_, and are consistent with the thresholded
Canny-style perimeter extraction used for tactical thermal-IR wildfire
edges in [mdpi2025tirfire]_.

**Limitations**, well documented in that literature: sensitivity to noise
and uneven illumination, per-experiment manual threshold tuning, failure
when flame and background have similar intensity (common in IR when the
background is also hot), and poor transfer across camera modalities
(visible vs. IR) or even across experiments with the same camera.

2.3 Machine learning approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Phase 2's goal is to replace or augment the row-by-row classical pipeline
with a model that generalises better across lighting, smoke, and modality.
The natural framing is **semantic segmentation**: classify every pixel as
"flame" or "not flame," producing a binary mask the same size as the input;
the row-wise edge ``x(y)`` is then just the boundary of that mask.

The one architecture concept worth knowing before you start researching:
**U-Net** [ronneberger2015unet]_ is the standard reference architecture for
this kind of pixel-wise segmentation task. It is an encoder that
progressively downsamples the image into a compact representation, paired
with a decoder that upsamples it back to full resolution, plus **skip
connections** that copy feature maps from each encoder stage directly to
the matching decoder stage. The skip connections are why U-Net is good at
*precise localisation* (sharp boundaries) even with a fairly small training
set — relevant here since a flame edge is exactly a "sharp boundary"
problem.

Beyond that starting point, this is intentionally something for you to
research rather than something to read off the page — the field moves fast
and the right architecture choice depends on details (frame rate, available
compute, how much annotated data we end up with) that aren't fixed yet.
Questions to investigate and bring back with your own findings:

- Has the **Segment Anything Model (SAM)** [kirillov2023sam]_ or its
  successors been applied to flame/fire segmentation? Does it work
  zero-shot, or does it need fine-tuning/adapters? Search for recent
  (2023–2025) papers on this specifically — flames are a visually unusual
  case for a model trained mostly on crisp, opaque object boundaries.
- What other segmentation architectures show up in recent (last 2–3 years)
  fire/flame/wildfire segmentation papers, and what tradeoffs do they
  report (accuracy, speed, dataset size needed)?
- Given what you find, what would *you* recommend as the first model to
  try for FlameTrack, and why? Come prepared to defend the choice — "what
  the papers did" isn't automatically "what fits our rig and dataset size."

2.4 Evaluation metrics
~~~~~~~~~~~~~~~~~~~~~~~~

Two complementary metrics, used together by recent fire-segmentation papers:

**Intersection over Union (IoU)**, for the predicted mask :math:`P` versus
ground-truth mask :math:`G` (both binary, same shape):

.. math::

   \text{IoU} = \frac{|P \cap G|}{|P \cup G|}

.. code-block:: python

   import numpy as np

   def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
       intersection = np.logical_and(pred_mask, gt_mask).sum()
       union = np.logical_or(pred_mask, gt_mask).sum()
       return intersection / union if union > 0 else 1.0

**Edge position error** — since FlameTrack's actual deliverable is the
row-wise edge ``x(y)``, not the mask itself, also report the mean absolute
error between predicted and ground-truth edge position, in pixels (and in
mm once the dewarping scale is known):

.. math::

   \text{MAE} = \frac{1}{H} \sum_{y=1}^{H} \left| x_{\text{pred}}(y) - x_{\text{true}}(y) \right|

.. code-block:: python

   def edge_mae(x_pred: np.ndarray, x_true: np.ndarray) -> float:
       return np.mean(np.abs(x_pred - x_true))

Compute both metrics for the **existing classical methods first** (fixed
threshold, gradient, Otsu) against a small hand-annotated set of frames —
this gives the fixed baseline that any ML model in Phase 2 must beat to be
worth the added complexity.

2.5 Data annotation
~~~~~~~~~~~~~~~~~~~~~

Any supervised model needs ground-truth labels: a binary flame/not-flame
mask (or boundary polyline) per annotated frame. Before deciding on a model,
you'll need a labelled dataset, which raises a few open questions worth
researching rather than assuming an answer to:

- Roughly how much annotated data does a model like U-Net realistically
  need to start producing useful results? Does this change much with data
  augmentation?
- What annotation tools exist for this kind of task, and which would fit
  best given we're annotating *video* (i.e. many frames from the same
  recording, often visually similar to their neighbours)? Look at general
  image-segmentation annotation tools as a starting point.
- What annotation strategy avoids wasting effort on near-duplicate frames
  (e.g. annotating every frame of a slow-moving flame front vs. sampling
  sparsely across many different experiments/conditions)?

Bring back a concrete proposal: which tool, roughly how many frames, and
how you'd sample them across the available recordings.

2.6 Connection: Phase 1 → Phase 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Emissivity-corrected frames (Phase 1) make pixel *temperature* values
physically comparable across regions and across experiments/specimens —
without correction, two experiments with different steel-frame oxidation
states would show different apparent temperatures for the *same* true
condition, adding spurious variance an ML model would otherwise have to
learn around. Feeding corrected frames into Phase 2 training should improve
both label consistency and model generalisation.

Data flow through the pipeline::

    Raw IR frames
        |
        v
    Dewarping                         (existing)
        |
        v
    Emissivity correction per ROI     (Phase 1, new)
        |
        v
    Flame edge detection              (Phase 2: classical baseline + ML)
        |
        v
    HDF5 results                      (existing)


References
----------

.. [planck1901] Planck, M. (1901). *On the Law of Distribution of Energy in
   the Normal Spectrum.* Annalen der Physik, 4, 553.

.. [incropera2007] Incropera, F. P., DeWitt, D. P., Bergman, T. L., &
   Lavine, A. S. (2007). *Fundamentals of Heat and Mass Transfer* (6th ed.).
   Wiley. Chapter 12: Radiation — Processes and Properties.

.. [otsu1979] Otsu, N. (1979). *A Threshold Selection Method from
   Gray-Level Histograms.* IEEE Transactions on Systems, Man, and
   Cybernetics, 9(1), 62–66.

.. [ronneberger2015unet] Ronneberger, O., Fischer, P., & Brox, T. (2015).
   *U-Net: Convolutional Networks for Biomedical Image Segmentation.*
   MICCAI 2015. https://arxiv.org/abs/1505.04597

.. [kirillov2023sam] Kirillov, A. et al. (2023). *Segment Anything.*
   ICCV 2023. https://arxiv.org/abs/2304.02643

.. [schartel2022twocolor] Schartel, B. et al. (2022). *Surface temperature
   and emissivity measurement for materials exposed to a flame through
   two-colour IR-thermography.* https://arxiv.org/pdf/2203.09689

.. [pitarma2019wood] Pitarma, R. et al. (2019). *An Approach Method to
   Evaluate Wood Emissivity.* Journal of Engineering, Wiley.
   https://onlinelibrary.wiley.com/doi/10.1155/2019/4925056

.. [fang2021building] Fang, et al. (2021). *Emissivity of Building
   Materials for Infrared Measurements.* PMC.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8002048/

.. [calcacademy] Calculator Academy. *Emissivity Correction Calculator.*
   https://calculator.academy/emissivity-correction-calculator/

.. [wildfireedge2025] Various authors (2024). *A comprehensive survey of
   research towards AI-enabled unmanned aerial systems in pre-, active-,
   and post-wildfire management.* https://arxiv.org/pdf/2401.02456

.. [mdpi2025tirfire] Various authors (2025). *Semi-Automated Extraction of
   Active Fire Edges from Tactical Infrared Observations of Wildfires.*
   Remote Sensing, MDPI. https://www.mdpi.com/2072-4292/17/21/3525
