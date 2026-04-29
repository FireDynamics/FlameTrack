Workflows
=========

Room Corner Experiment
-----------------------

A room corner test measures fire spread across two perpendicular wall panels.

1. Load IR data from the ``exported_data/`` folder
2. Select **Room Corner** as experiment type
3. Place 6 calibration points — 3 on the left panel, 3 on the right
4. Enter plate dimensions and run **Dewarp**
5. Run **Find Edge** for both left and right panels (multithreaded)
6. Results are stored in HDF5 under ``edge_results_left`` and ``edge_results_right``

Lateral Flame Spread
---------------------

A lateral flame spread test measures horizontal fire spread on a single panel.

1. Load data (IR, video, or images)
2. Select **Lateral Flame Spread**
3. Place 4 corner points
4. Run **Dewarp**, then **Find Edge**
5. Results are stored in HDF5 under ``edge_results``

Output Format
-------------

All results are stored in an HDF5 file (``<experiment_name>_results_RCE.h5``)
in the ``processed_data/`` subfolder:

.. code-block:: text

    experiment_results_RCE.h5
    ├── dewarped_data_left/
    │   ├── data            # float32 array (H, W, T)
    │   └── attrs           # transformation_matrix, plate dimensions, ...
    ├── dewarped_data_right/
    │   └── ...
    ├── edge_results_left/
    │   └── data            # int array (T, H) — edge x-position per row per frame
    └── edge_results_right/
        └── ...

The file also stores ``flametrack_version`` and ``flametrack_commit`` as root
attributes for full reproducibility.
