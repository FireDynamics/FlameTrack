Configuration
=============

No configuration is needed to get started. Launch FlameTrack, click
**Open Folder**, and navigate to your experiment directory.

Expected folder structure
--------------------------

FlameTrack expects IR data in an ``exported_data/`` sub-folder and writes
results to ``processed_data/`` (created automatically):

.. code-block:: text

    my_experiment/
    ├── exported_data/        ← CSV files exported from the IR camera
    │   ├── frame_0000.csv
    │   ├── frame_0001.csv
    │   └── ...
    └── processed_data/       ← created by FlameTrack
        └── my_experiment_results_RCE.h5

For video or image sequences, place the file(s) directly in the experiment
folder — FlameTrack detects the data type automatically.

.. note::

   **Advanced:** FlameTrack reads a ``config.ini`` file that can set a default
   ``experiment_folder`` so that a list of experiments is pre-populated on
   startup. The file is created automatically next to the installed package.
   You only need to edit it if you want that convenience — most users never
   touch it.
