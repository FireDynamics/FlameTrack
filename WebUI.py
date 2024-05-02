from dash import Dash, Input, Output, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import glob
import numpy as np
import os
from DataTypes import VideoData, IrData
import progressbar
from IR_analysis import sort_corner_points, dewarp_data
from collections import deque
import user_config

# Parameters for the dewarping
TARGET_RATIO = 15.5/80 #ratio of width to height


# Measurment name is used for saving the data
# Data structure is: {data}/{MEASUREMENT_NAME}/file1.csv, file2.csv, ...
# Path where the data folder is located in config.ini

MEASUREMENT_NAME = 'lfs_pmma_DE_6mm_tc_R2_0001'

#Hier wird entschieden um was für Daten es sich handelt
data = IrData(os.path.join(user_config.get_path('data_folder'), MEASUREMENT_NAME))







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
raw_data_fig.update_layout({'title': 'Original Data'})
dewarped_data_fig.update_layout({'title': 'Dewarped Data'})


def update_dewarped_figure(img):
    global TARGET_RATIO
    global dewarped_data_fig
    global selected_points
    if len(selected_points) != 4:
        return
    dewarped_data = dewarp_data(img, sort_corner_points(selected_points), target_ratio=TARGET_RATIO)
    dewarped_data_fig.update_layout({'width': img_height/TARGET_RATIO ,
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
            dcc.Slider(id='data-slider', min=data_numbers[0], max=data_numbers[-1], value=data_numbers[0]),
            html.P('Select min temperature'),
            dcc.Slider(id='min-temp-slider', min=int(img_min), max=int(img_max), value=int(img_min)),
            html.P('Select max temperature'),
            dcc.Slider(id='max-temp-slider', min=int(img_min), max=int(img_max), value=int(img_max)),
        ],
        ),
        dbc.Col([
            dcc.Graph(id=dewarped_data_plot_id, figure=dewarped_data_fig),
            dcc.Loading(
                id="loading",
                type="default",  # or "cube", "circle", "dot", "cube"

                children=[
                    dbc.Button("Save Data", id="save-button", color="primary", className="mr-1"),  # new button
                ]
            ),
            dcc.RangeSlider(
                id='range-slider',
                min=data_numbers[0],
                max=data_numbers[-1],
                step=1,),

        ]),
    ])
], fluid=True)


# Callback for saving the data
@app.callback(
    Output('save-button', 'children'),  # this output is dummy, just to allow the callback
    Input('save-button', 'n_clicks'),
    Input('range-slider', 'value'),
    prevent_initial_call=True,
)
def save_data(n_clicks,frame_range):
    global data_files
    global selected_points
    global TARGET_PIXELS_WIDTH, TARGET_PIXELS_HEIGHT
    print(frame_range)
    if n_clicks is None or len(selected_points) != 4:
        raise PreventUpdate
    else:
        bar = progressbar.ProgressBar()
        start,end = frame_range
        result = None
        for idx in bar(data_numbers [start:end]):
            img = data.get_frame(idx)
            dewarped_data = dewarp_data(img, sort_corner_points(selected_points), target_ratio=TARGET_RATIO)
            # os.makedirs(f'dewarped_data/{MEASUREMENT_NAME}', exist_ok=True)
            # np.savetxt(f'dewarped_data/{MEASUREMENT_NAME}/{os.path.basename(file)}', dewarped_data, delimiter=';')
            if result is None:
                result = dewarped_data
            else:
                result = np.dstack((result, dewarped_data))
        dewarped_data_path=user_config.get_path('dewarped_data')
        os.makedirs('dewarped_data_path', exist_ok=True)
        np.save(f'{dewarped_data_path}/{MEASUREMENT_NAME}_dewarped.npy', result)
        # Show save confirmation
        dbc.Toast(
            "Data saved!",
            header="Save Confirmation",
            icon="success",
            dismissable=True,
            duration=5000,
        )
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
