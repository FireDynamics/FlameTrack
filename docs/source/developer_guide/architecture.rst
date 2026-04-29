Architecture Overview
=====================

FlameTrack is structured in three layers:

.. code-block:: text

    ┌─────────────────────────────────────────┐
    │              GUI Layer                  │
    │  main_window · imshow_canvas ·          │
    │  selectable_imshow_canvas ·             │
    │  draggable_point · plotting_utils       │
    ├─────────────────────────────────────────┤
    │           Processing Layer              │
    │  dewarping · edge_worker                │
    ├─────────────────────────────────────────┤
    │            Analysis Layer               │
    │  flamespread · ir_analysis ·            │
    │  dataset_handler · data_types           │
    └─────────────────────────────────────────┘

Module Overview
---------------

**flametrack.gui**

``main_window``
    Central Qt window. Owns all UI state, starts worker threads for
    dewarping and edge detection, writes results to HDF5.

``selectable_imshow_canvas``
    Extends ``ImshowCanvas`` with interactive draggable calibration points.
    Handles mouse clicks and keyboard shortcuts (D = delete, C = clear).

``imshow_canvas``
    PyQtGraph-based image display widget with adjustable color range.

``draggable_point``
    Single draggable marker on the canvas. Notifies the parent canvas on move.

``plotting_utils``
    Stateless helpers: ``sort_corner_points``, ``rotate_points``.
    Used to map clicked coordinates back to unrotated image space.

**flametrack.processing**

``dewarping``
    ``dewarp_room_corner_remap`` and ``dewarp_lateral_flame_spread`` — apply
    homography-based perspective correction and write dewarped frames to HDF5.
    ``DewarpConfig`` dataclass holds all parameters.

**flametrack.analysis**

``flamespread``
    Edge detection functions and the ``EDGE_METHOD_CATALOG`` (six variants).
    ``calculate_edge_data`` is the main entry point: applies an ``EdgeFn``
    row-by-row across all frames, optionally with Otsu masking.

``ir_analysis``
    Low-level helpers: ``read_ir_data`` (parse IR CSV), ``get_dewarp_parameters``
    (compute homography from 4 corner points), ``compute_remap_from_homography``.

``data_types``
    ``IrData``, ``VideoData``, ``ImageData`` — unified frame-access interface.
    ``RceExperiment`` — manages the folder structure and lazy-loads the HDF5 file.

``dataset_handler``
    HDF5 file creation, schema initialization, and data access helpers.

**flametrack.utils**

``math_utils``
    ``compute_target_ratio``, ``estimate_resolution_from_points``.

Data Flow
---------

.. mermaid::

   flowchart TD
       A([User clicks Dewarp]) --> B[main_window\nreads points + DewarpConfig]
       B --> C[dewarping.dewarp_room_corner_remap]
       C --> D[ir_analysis.get_dewarp_parameters\ncompute homography]
       D --> E[cv2.remap per frame]
       E --> F[(HDF5\ndewarped_data_left/right)]

       G([User clicks Find Edge]) --> H[main_window\ncreates EdgeDetectionWorker]
       H --> I[worker.run — QThread]
       I --> J[flamespread.calculate_edge_data]
       J --> K{Otsu masking\nenabled?}
       K -- yes --> L[restrict search window\nper row]
       K -- no --> M[full row]
       L --> N[EdgeFn row-by-row]
       M --> N
       N --> O[main_window.handle_edge_result]
       O --> P[(HDF5\nedge_results_left/right)]
