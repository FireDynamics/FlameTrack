from datetime import datetime

from dash import Dash, Input, Output, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import glob
import numpy as np
import os
from DataTypes import VideoData, IrData,ImageData
import progressbar
from IR_analysis import sort_corner_points, dewarp_data, get_dewarp_parameters
from collections import deque
from dataset_handler import *
import dataset_handler
import user_config

# Parameters for the dewarping
TARGET_RATIO = 15.5 / 80  # ratio of width to height

# Measurment name is used for saving the data
# Data structure is: {data}/{MEASUREMENT_NAME}/file1.csv, file2.csv, ...
# Path where the data folder is located in config.ini


exp_names = [f'lfs_pmma_DE_6mm_tc_R{number}_{datatype}' for datatype in ['CANON', 'IR'] for number in [1,2, 3, 4]]
MEASUREMENT_NAME = exp_names[3] # <--- Change this to the measurement you want to dewarp
# exp_names[0] = 'lfs_pmma_DE_6mm_tc_R1_CANON'
# exp_names[1] = 'lfs_pmma_DE_6mm_tc_R2_CANON'
# exp_names[2] = 'lfs_pmma_DE_6mm_tc_R3_CANON'
# exp_names[3] = 'lfs_pmma_DE_6mm_tc_R4_CANON'
# exp_names[4] = 'lfs_pmma_DE_6mm_tc_R1_IR'
# exp_names[5] = 'lfs_pmma_DE_6mm_tc_R2_IR'
# exp_names[6] = 'lfs_pmma_DE_6mm_tc_R3_IR'
# exp_names[7] = 'lfs_pmma_DE_6mm_tc_R4_IR'






#
if 'IR' in MEASUREMENT_NAME:
    data = IrData(os.path.join(user_config.get_path('data_folder'), MEASUREMENT_NAME.replace('IR', "0001")))
else:
    data = ImageData(os.path.join(user_config.get_path('canon_folder'), MEASUREMENT_NAME.replace('_CANON', "")), 'JPG')

selected_points = deque(maxlen=4)
plasma_colors = px.colors.sequential.Plasma
colorscale = [[i / (len(plasma_colors) - 1), color] for i, color in enumerate(plasma_colors)]
data_numbers = data.data_numbers
# open image and create plotly figure, locally stored images can be used this way too
img = data.get_frame(data_numbers[0])
img_width, img_height = img.shape[1], img.shape[0]
img_min, img_max = np.nanmin(img), np.nanmax(img)
raw_data_fig = go.Figure()
dewarped_data_fig = go.Figure()

raw_data_fig.add_trace({
    'z': img,
    'type': 'heatmap',
    'name': 'IR_data',
    'colorscale': colorscale
})

# update figure layout, actually not necessary for the functionality
figure_layout = {
    'template': 'plotly_dark',
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'width': img_width * 1,
    'height': img_height * 1,
    'xaxis': {'showgrid': False,
              'showticklabels': False
              },
    'yaxis': {'showgrid': False,
              'showticklabels': False
              },

}
raw_data_fig.update_layout(figure_layout)
dewarped_data_fig.update_layout(figure_layout)
raw_data_fig.update_layout({'title': f'Original Data {MEASUREMENT_NAME}'})
dewarped_data_fig.update_layout({'title': 'Dewarped Data'})


def update_dewarped_figure(img):
    global TARGET_RATIO
    global dewarped_data_fig
    global selected_points
    if len(selected_points) != 4:
        return
    dewarped_params = get_dewarp_parameters(sort_corner_points(selected_points), target_ratio=TARGET_RATIO)
    dewarped_data = dewarp_data(img, dewarped_params)
    dewarped_data_fig.update_layout({'width': img_height / TARGET_RATIO,
                                     'height': img_height,
                                     'title': f'Dewarped Data ({dewarped_data.shape[1]}x{dewarped_data.shape[0]})'})
    dewarped_data_fig.data = []
    dewarped_data_fig.add_trace({
        'z': dewarped_data,
        'type': 'heatmap',
        'name': 'dewarped data',
    })


# initiate app, use a dbc theme
app = Dash(__name__,
           external_stylesheets=[dbc.themes.SLATE],
           meta_tags=[
               {'name': 'viewport',
                'content': 'width=device-width, initial-scale=1.0'
                }
           ]
           )

raw_data_plot_id = 'raw_data_plot'
dewarped_data_plot_id = 'dewarped_data_plot'
app.layout = dbc.Container([
    html.H4('IR Data Dewarping'),
    html.P('Click on the image to select the corners of the area you want to dewarp'),
    html.P('### Instructions ###'),
    html.P('drag for zooming'),
    html.P('SHIFT + drag to pan'),
    dbc.Row([
        dbc.Col([

            dcc.Graph(id=raw_data_plot_id, figure=raw_data_fig),

            html.P('Select the file number to display'),
            dcc.Slider(id='data-slider', min=data_numbers[0], max=data_numbers[-1], value=data_numbers[0],step=1),
            html.P('Select min temperature'),
            dcc.Slider(id='min-temp-slider', min=int(img_min), max=int(img_max), value=int(img_min)),
            html.P('Select max temperature'),
            dcc.Slider(id='max-temp-slider', min=int(img_min), max=int(img_max), value=int(img_max)),
        ],
        ),
        dbc.Col([
            dcc.Graph(id=dewarped_data_plot_id, figure=dewarped_data_fig),

            html.P('Select the range of frames to dewarp'),
            dbc.Row([
                dbc.Col([
                    html.P('Start frame'),
                    dcc.Input(id='range-start', type='number', value=data_numbers[0]),
                ]),
                dbc.Col([
                    html.P('End frame'),
                    dcc.Input(id='range-end', type='number', value=data_numbers[-1]),
                ]),
            ]),
            dcc.Loading(
                id="loading",
                type="default",  # or "cube", "circle", "dot", "cube"

                children=[
                    dbc.Button("Save Data", id="save-button", color="primary", className="mr-1"),  # new button
                ]
            ),

        ]),
    ])
], fluid=True)



# Callback for saving the data
@app.callback(
    Output('save-button', 'children'),  # this output is dummy, just to allow the callback
    Input('save-button', 'n_clicks'),
    Input('range-start', 'value'),
    Input('range-end', 'value'),
    prevent_initial_call=True,
)
def save_data(n_clicks, frame_range_start, frame_range_end):
    global data_files
    global selected_points
    global TARGET_PIXELS_WIDTH, TARGET_PIXELS_HEIGHT
    if n_clicks is None or len(selected_points) != 4:
        raise PreventUpdate
    else:

        start, end = frame_range_start, frame_range_end
        dewarp_params = get_dewarp_parameters(sort_corner_points(selected_points), target_ratio=TARGET_RATIO)

        h5_file = create_h5_file(f'{MEASUREMENT_NAME}')
        grp = h5_file['dewarped_data']
        grp.attrs['transformation_matrix'] = dewarp_params['transformation_matrix']
        grp.attrs['target_pixels_width'] = dewarp_params['target_pixels_width']
        grp.attrs['target_pixels_height'] = dewarp_params['target_pixels_height']
        grp.attrs['target_ratio'] = TARGET_RATIO
        grp.attrs['selected_points'] = selected_points
        grp.attrs['frame_range'] = (start, end)
        grp.attrs['points_selection_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dset_h,dset_w = dewarp_params['target_pixels_height'], dewarp_params['target_pixels_width']

        dset = grp.create_dataset('data',
                  (dset_h, dset_h, 1),
                    maxshape=(dset_h, dset_w, None),
                    chunks=(dset_h, dset_w, 1),

                           dtype=np.float32)
        # import time
        # start_time = time.time()
        # print('Dewarping data')
        # for i,idx in enumerate(data_numbers[start:end]):
        #     img = data.get_frame(idx)
        #     dewarped_data = dewarp_data(img, dewarp_params)
        #     dset.resize((dset_h, dset_w, i + 1))
        #     dset[:, :, i] = dewarped_data
        #     if idx % 100 == 0:
        #         print(f'Estimated time left: {((time.time() - start_time) / i) * (end-start - i)} seconds')
        # dataset_handler.close_file()
        # dbc.Toast(
        #     "Data saved!",
        #     header="Save Confirmation",
        #     icon="success",
        #     dismissable=True,
        #     duration=5000,
        # )
    return "Save Data"


@app.callback(
    Output(raw_data_plot_id, 'figure', allow_duplicate=True),
    Input('min-temp-slider', 'value'),
    Input('max-temp-slider', 'value'),
    prevent_initial_call=True,
)
def update_colorscale(min_temp, max_temp):
    global raw_data_fig
    global colorscale
    global img_min, img_max

    min_data, max_data = img_min, img_max
    min_scale = max((min_temp - min_data) / (max_data - min_data), 0)
    min_scale = min(min_scale, 1)
    max_scale = min((max_temp - min_data) / (max_data - min_data), 1)
    max_scale = max(max_scale, 0)
    scales = np.linspace(min_scale, max_scale, len(plasma_colors))
    scales = list(scales)
    colorscale = [[scales[i], color] for i, color in enumerate(plasma_colors)]
    colorscale = [[0, plasma_colors[0]]] + colorscale + [[1, plasma_colors[-1]]]
    for trace in raw_data_fig.data:
        if trace.name == 'IR_data':
            trace.colorscale = colorscale
            break
    return raw_data_fig


@app.callback(
    Output(raw_data_plot_id, 'figure', allow_duplicate=True),
    Output(dewarped_data_plot_id, 'figure', allow_duplicate=True),
    Output('min-temp-slider', 'min'),
    Output('min-temp-slider', 'max'),
    Output('min-temp-slider', 'value'),
    Output('max-temp-slider', 'min'),
    Output('max-temp-slider', 'max'),
    Output('max-temp-slider', 'value'),
    Input('data-slider', 'value'),
    prevent_initial_call=True,
)
def update_figure(data_index):
    global raw_data_fig
    global dewarped_data_fig  # make sure to use the global fig variable
    global img, img_min, img_max  # make sure to use the global img_min and img_max variables
    if data_index not in data_numbers:
        raise PreventUpdate
    img = data.get_frame(data_index)
    img_min, img_max = np.nanmin(img), np.nanmax(img)
    update_dewarped_figure(img)
    for trace in raw_data_fig.data:
        if trace.name == 'IR_data':
            trace.z = img
            return raw_data_fig, dewarped_data_fig, img_min, img_max, img_min, img_min, img_max, img_max


@app.callback(
    Output(raw_data_plot_id, 'figure', allow_duplicate=True),
    Output(dewarped_data_plot_id, 'figure', allow_duplicate=True),
    Input(raw_data_plot_id, 'clickData'),
    prevent_initial_call=True,

)
def get_click(clickData):
    global raw_data_fig  # make sure to use the global fig variable
    global selected_points  # make sure to use the global selected_points variable
    global dewarped_data_fig
    global img
    if not clickData:
        raise PreventUpdate
    else:
        points = clickData.get('points')[0]
        x = points.get('x')
        y = points.get('y')
        selected_points.append([x, y])
        sorted_points = sort_corner_points(selected_points)

    # create a new trace for the selected points
    selected_trace = {
        'x': [point[0] for point in sorted_points] + [sorted_points[0][0]],
        'y': [point[1] for point in sorted_points] + [sorted_points[0][1]],
        # dottet line
        'mode': 'lines+markers',
        'line': {'dash': 'dot'},
        'marker': {'size': 10, 'color': 'red'},
        'name': 'selected points'
    }
    # remove the old trace from the figure
    raw_data_fig.data = [trace for trace in raw_data_fig.data if trace.name != 'selected points']
    # add the new trace to the figure
    raw_data_fig.add_trace(selected_trace)

    update_dewarped_figure(img)

    return raw_data_fig, dewarped_data_fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8052)
