import os
import numpy as np
import scipy
import streamlit as st

import user_config
from flamespread import show_flame_spread,show_flame_contour,load_data
st.set_page_config(layout="wide")
import streamlit.components.v1 as components

st.title('Framespread Analysis')

dewarped_data_folder = user_config.get_path('dewarped_data_folder')
edge_results_folder = user_config.get_path('edge_results_folder')

dewarped_data_files = os.listdir(dewarped_data_folder)
edge_results_files = os.listdir(edge_results_folder)

data_form = st.form(key='data_form')
with data_form:
    dewarped_data_selected = st.selectbox('Select dewarped data', dewarped_data_files)
    edge_results_selected = st.selectbox('Select edge results', edge_results_files)
    rolling_window = st.slider('Select rolling window', 1, 51, 1,2)
    data_submitted =st.form_submit_button('Load data')
if data_submitted:

    dewarped_data = load_data(f'{dewarped_data_folder}/{dewarped_data_selected}')
    edge_results = load_data(f'{edge_results_folder}/{edge_results_selected}')
    st.session_state['dewarped_data'] = dewarped_data
    st.session_state['edge_results'] = edge_results

    if rolling_window > 1:
        edge_results_rolled = np.array(scipy.signal.medfilt(edge_results,rolling_window))
        st.session_state['edge_results'] = edge_results_rolled



dewarped_data = st.session_state.get('dewarped_data',None)
edge_results = st.session_state.get('edge_results',None)

if dewarped_data is not  None and edge_results is not None:
    st.write(f'Data loaded from {dewarped_data_selected} and {edge_results_selected} with rolling window {rolling_window}')

    c1,c2 = st.columns([1,1])

    with c1:
        frame = st.slider('Select frame', 0, dewarped_data.shape[-1]-1, 0)
    with c2:
        y_coord = st.slider('Select y coordinate', 0, dewarped_data.shape[0]-1, 0)
        fig,ax = show_flame_spread(edge_results, y_coord)
        ax.axvline(frame, color='green', linestyle='--')
        ax.axhline(edge_results[frame,-y_coord-1], color='green', linestyle='--')
        st.pyplot(fig)

    with c1:
        fig, ax = show_flame_contour(dewarped_data, edge_results, frame)
        ax.axhline(y_coord, color='green', linestyle='--')
        ax.axvline(edge_results[frame,-y_coord-1], color='green', linestyle='--')
        st.pyplot(fig)