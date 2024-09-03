import os

import plotly.express as px

import user_config
from dataset_handler import get_dewarped_metadata, close_file
from DataTypes import IrData,ImageData
import streamlit as st



exp_names = os.listdir(user_config.get_path('processed_data'))
exp_names.sort()
exp_names = [exp_name for exp_name in exp_names if exp_name.endswith('.h5')]
exp_names = [exp_name[:-3] for exp_name in exp_names ]
data_form = st.form(key='data_form')
with data_form:
    exp_name_selected = st.selectbox('Select Experiment', exp_names )
    data_submitted =st.form_submit_button('Load data')
if data_submitted:

    dewarped_meta_data = dict(get_dewarped_metadata(exp_name_selected))
    close_file()
    st.session_state['dewarped_meta_data'] = dewarped_meta_data
    if 'CANON' in exp_name_selected:
        data = ImageData(os.path.join(r'/Volumes/Tam Backup/OM/', exp_name_selected.replace('_CANON',"")), 'JPG')
    else:
        data = IrData(os.path.join(user_config.get_path('data_folder'), exp_name_selected))
    st.session_state['data'] = data

dewarped_meta_data = st.session_state.get('dewarped_meta_data',None)
data = st.session_state.get('data',None)


if dewarped_meta_data is not  None and data is not None:
    frame_nr = st.slider('Select frame', 0, data.get_frame_count()-1, 0)
    frame = data.get_frame(frame_nr)
    fig = px.imshow(frame, color_continuous_scale='plasma')
    selected_points = dewarped_meta_data['selected_points']
    fig.add_trace(px.scatter(x=[p[0] for p in selected_points], y=[p[1] for p in selected_points]).data[0])
    st.plotly_chart(fig)
