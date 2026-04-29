Quickstart
==========

This guide walks through a typical FlameTrack session from loading data to exporting results.

1. Launch FlameTrack
---------------------

::

    flametrack

2. Select Experiment Type
--------------------------

Use the dropdown to choose between:

- **Room Corner** – two-sided setup, requires 6 calibration points
- **Lateral Flame Spread** – single panel, requires 4 calibration points

3. Load Data
------------

Click **Open Folder** and select the folder containing your experiment data.
FlameTrack auto-detects the data type (IR CSV, video, or images).

4. Set Rotation
---------------

If the camera was mounted sideways, select the matching rotation from the
**Rotation** dropdown before setting calibration points.

5. Set Calibration Points
--------------------------

Click directly on the image to place the required corner points.
Points must follow the order shown in the diagram for your experiment type.

- Press **D** to delete the nearest point
- Press **C** to clear all points

6. Enter Plate Dimensions
--------------------------

Enter the physical width and height of the specimen plate in millimeters.
FlameTrack uses these to compute the target image resolution.

7. Run Dewarping
-----------------

Click **Dewarp**. Results are saved as an HDF5 file in the
``processed_data/`` subfolder of your experiment directory.

8. Run Edge Detection
----------------------

Select the flame direction (left-to-right or right-to-left), adjust the
threshold slider if needed, and click **Find Edge**.

9. Inspect Results
-------------------

Use the **Analysis Y** slider to inspect the detected edge at different
heights. The edge position over time is shown in the analysis plot.
