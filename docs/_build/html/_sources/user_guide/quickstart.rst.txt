Quickstart
==========

This guide walks through a complete FlameTrack session from loading data to
saving results.

Before you start, make sure your experiment folder follows the expected
structure — see :doc:`configuration`.

1. Launch FlameTrack
---------------------

::

    flametrack

2. Select Experiment Type
--------------------------

Use the **Experiment type** dropdown:

- **Lateral Flame Spread (LFS)** — single panel, 4 calibration points
- **Room Corner** — two perpendicular panels, 6 calibration points

3. Load Data
------------

Click **Open Folder** and select your experiment. FlameTrack loads the first
frame and displays it.

.. figure:: ../_static/screenshots/quickstart_loaded.png
   :alt: FlameTrack after opening a folder — first IR frame displayed
   :width: 90%

   After opening a folder: the first thermal IR frame fills the image area.
   No calibration points placed yet.

4. Set Camera Rotation
-----------------------

Look at the loaded image. If the plate appears rotated, choose the matching
angle from the **Rotation** dropdown — the image rotates immediately. Set
this *before* placing calibration points.

5. Place Calibration Points
----------------------------

Click on the four (or six) corners of the specimen plate. The **click order
does not matter** — FlameTrack sorts the points automatically.

**Lateral Flame Spread — 4 points**

Click the four corners of the plate:

.. code-block:: text

    TL ─────────────── TR
    │                   │
    │   specimen plate  │
    │                   │
    BL ─────────────── BR

    TL = top-left   TR = top-right
    BL = bottom-left  BR = bottom-right

**Room Corner — 6 points**

Click the three corners of the left panel and the three corners of the right
panel (the inner vertical edge is shared):

.. code-block:: text

    L-TL      L-TR / R-TL      R-TR
      │    left  │    right  │
      │  panel   │   panel   │
    L-BL      L-BR / R-BL      R-BR

    Place one point at each marked corner (6 total).
    The two panels share the inner vertical edge.

.. figure:: ../_static/screenshots/quickstart_calibration_roomcorner.png
   :alt: Six calibration points placed on a Room Corner experiment
   :width: 90%

   Six calibration points placed on a Room Corner frame. The lines connect
   the three corners of each panel, meeting at the shared inner edge.

**Keyboard shortcuts while placing points**

- **D** — delete the point nearest to the cursor
- **C** — clear all points

6. Enter Plate Dimensions
--------------------------

Enter the physical width and height of the specimen in **millimetres**.
FlameTrack uses these values to compute the output image resolution.

7. Run Dewarping
-----------------

Click **Dewarp**. FlameTrack corrects perspective distortion for every frame
and saves the result to::

    processed_data/<experiment_name>_results_RCE.h5

A progress bar shows how many frames have been processed.

8. Run Edge Detection
----------------------

Select the flame direction (**left → right** or **right → left**) from the
**Flame direction** dropdown, then click **Find Edge**.

For a Room Corner experiment both panels are processed simultaneously in
separate threads.

.. figure:: ../_static/screenshots/quickstart_edge_result.png
   :alt: Dewarped frame with detected flame edge overlaid
   :width: 90%

   Edge recognition result: the dewarped plate fills the image, and the
   detected flame edge is overlaid as a cyan line.

9. Inspect Results
-------------------

Use the **Analysis Y** slider to select a horizontal row. The lower plot shows
the detected flame-edge position at that height over time.

.. figure:: ../_static/screenshots/readme_spread.png
   :alt: Flame spread curve — edge position over time for left and right panel
   :width: 90%

   Analysis view: flame edge position over time at the selected height.
   For a Room Corner experiment both panels (left = red, right = blue mirrored)
   are shown with a confidence band across all rows.

The full result array is stored in the HDF5 file under ``edge_results``
(LFS) or ``edge_results_left`` / ``edge_results_right`` (Room Corner).
See :doc:`workflows` for the complete output format.
