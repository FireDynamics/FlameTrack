import os
import numpy as np
import scipy
import streamlit as st

import user_config
from flamespread import show_flame_spread,show_flame_contour,plot_gradient,get_frame
from dataset_handler import get_dewarped_data, get_edge_results
st.set_page_config(layout="wide")
import streamlit.components.v1 as components

st.title('Framespread Analysis')


exp_names = os.listdir(user_config.get_path('saved_data'))
exp_names = [exp_name[:-3] for exp_name in exp_names ]
data_form = st.form(key='data_form')
with data_form:
    exp_name_selected = st.selectbox('Select Experiment', exp_names )
    rolling_window = st.slider('Select rolling window', 1, 51, 1,2)
    data_submitted =st.form_submit_button('Load data')
if data_submitted:

    dewarped_data = get_dewarped_data(exp_name_selected)
    edge_results = get_edge_results(exp_name_selected)[:]
    st.session_state['dewarped_data'] = dewarped_data
    st.session_state['edge_results'] = edge_results

    if rolling_window > 1:
        edge_results_rolled = np.array(scipy.signal.medfilt(edge_results,rolling_window))
        st.session_state['edge_results'] = edge_results_rolled



dewarped_data = st.session_state.get('dewarped_data',None)
edge_results = st.session_state.get('edge_results',None)

if dewarped_data is not  None and edge_results is not None:

    c1,c2 = st.columns([1,1])

    with c1:
        frame = st.slider('Select frame', 0, dewarped_data.shape[-1]-1, 0)
    with c2:
        y_coord = st.slider('Select y coordinate', 0, dewarped_data.shape[0]-1, 0)
        fig,ax = show_flame_spread(edge_results, -y_coord)
        ax.axvline(frame, color='green', linestyle='--')
        ax.axhline(edge_results[frame,y_coord], color='green', linestyle='--')
        st.pyplot(fig)

    with c1:
        fig, ax = show_flame_contour(dewarped_data, edge_results, frame)
        h= ax.get_ylim()[1]
        ax.axhline(y_coord, color='green', linestyle='--')
        ax.axvline(edge_results[frame,y_coord], color='green', linestyle='--')
        st.pyplot(fig)
        fig,ax = plot_gradient(get_frame(dewarped_data,frame),y_coord)
        ax.axvline(edge_results[frame, y_coord], color='green', linestyle='--')
        st.pyplot(fig)