# FireTrack

#Installation
1. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```


#Instructions
1. To dewarp the image, run WebUi.py. Select the points and then press save. This creates an h5 file that has the correct dewarp parameters. To select the right experiment edit the first lines in WebUi.py and configure config.ini correspondingly
2. After that run dewarp_manual.py. Select the right experiment in the python code
3. Run streamlit_analysis.py by running `streamlit run streamlit_analysis.py` in the terminal. This will open a web interface where you can select the experiment to check if the flamespread calculation is correct.


# Alternate RCE Experiments
1. Run QTGUI.py
2. Press load and select the experiment folder. The fodler structure must be "exp_name"/exported_data/ and "exp_name"/processed_data/
3. Go to the edge results tab and press the button
4. check the results in the flamespread tab. If no lines are shown try to move the slider.
