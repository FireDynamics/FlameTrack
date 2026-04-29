Workflows
=========

Room Corner Experiment
-----------------------

A room corner test measures fire spread across two perpendicular wall panels.

1. Set ``experiment_folder`` in ``config.ini`` and launch FlameTrack
2. Select **Room Corner** as experiment type
3. Load the experiment folder
4. Place 6 calibration points — 3 on the left panel, 3 on the right
5. Enter plate dimensions and click **Dewarp**
6. Click **Find Edge** — both panels are processed in parallel
7. Results are stored in the HDF5 file (see below)

Lateral Flame Spread
---------------------

A lateral flame spread test measures horizontal fire spread on a single panel.

1. Load the experiment folder
2. Select **Lateral Flame Spread**
3. Place 4 corner points on the specimen plate
4. Enter plate dimensions and click **Dewarp**
5. Select flame direction and click **Find Edge**
6. Results are stored in the HDF5 file (see below)

Output Format
-------------

All results are saved to an HDF5 file in the ``processed_data/`` sub-folder
of the experiment directory.

**Lateral Flame Spread**

File: ``<experiment_name>_results_RCE.h5``

.. code-block:: text

    <experiment_name>_results_RCE.h5          (root attributes below)
    │  flametrack_version                      e.g. "1.0.0"
    │  flametrack_commit                       e.g. "a3f9c12"
    │  plate_width_mm
    │  plate_height_mm
    ├── dewarped_data/
    │   ├── data                               float32 array  (frames, H, W)
    │   └── attrs: transformation_matrix, plate_width_mm, plate_height_mm, …
    └── edge_results/
        └── data                               int array  (frames, H)
                                               edge x-position per row per frame

**Room Corner**

File: ``<experiment_name>_results_RCE.h5``

.. code-block:: text

    <experiment_name>_results_RCE.h5
    │  flametrack_version  /  flametrack_commit
    │  plate_width_mm_left  /  plate_height_mm_left
    │  plate_width_mm_right /  plate_height_mm_right
    ├── dewarped_data_left/
    │   ├── data                               float32 array  (frames, H, W)
    │   └── attrs: transformation_matrix, …
    ├── dewarped_data_right/
    │   └── …
    ├── edge_results_left/
    │   └── data                               int array  (frames, H)
    └── edge_results_right/
        └── data                               int array  (frames, H)

.. figure:: ../_static/screenshots/readme_pipeline.png
   :alt: Room Corner pipeline — calibration, dewarped panels and detected edges
   :width: 100%

   Room Corner experiment: calibration points on the raw IR frame (left),
   dewarped panels with detected edges for left and right wall (right).

.. figure:: ../_static/screenshots/readme_spread.png
   :alt: Flame spread curve for a Room Corner experiment
   :width: 100%

   Analysis view for the same experiment: flame edge position over time.
   Left panel (red) and right panel (blue, mirrored) with confidence band.

Reading results in Python
--------------------------

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File("my_experiment_results_RCE.h5", "r") as f:
        print(f.attrs["flametrack_version"])   # version used for analysis
        edge = f["edge_results"]["data"][:]    # shape: (frames, rows)
        dewarped = f["dewarped_data"]["data"]  # lazy HDF5 dataset

    # edge[t, y] = x-position of the flame edge at row y in frame t
    # -1 indicates no edge detected for that row/frame
