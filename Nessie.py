import dash_bootstrap_components as dbc
from dash import dash, Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import numpy as np
import plotly.express as px
import os
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from io import StringIO
import re
from data_loader import GliderDataLoader, GulfStreamLoader, MapDataLoader, GliderGridDataLoader, MPADataLoader, gomofsdataloader, doppiodataloader
# from nessie_interpolation_function import create_spatial_interpolation
import datetime as dt
from typing import cast, List, Dict, Any
import pytz
from plotly.validator_cache import ValidatorCache
import numpy as np
# from scipy.interpolate import griddata

def get_first_10_pH_average(df_latest):
    df_MLD_average = df_latest.drop_duplicates(subset=['Station', 'Cruise'], keep='first').copy()

    # Initialize new columns
    df_MLD_average['pHinsitu[Total]'] = np.nan
    df_MLD_average['Chl_a[mg/m^3]'] = np.nan

    # Group by both Station and Cruise
    for (station, cruise), group in df_latest.groupby(['Station', 'Cruise']):
        first_10 = group.head(5)
        
        if 'pHinsitu[Total]' in first_10.columns:
            avg_pH = first_10['pHinsitu[Total]'].mean()
            avg_chl = first_10['Chl_a[mg/m^3]'].mean()
            
            # Add directly to DataFrame using both Station and Cruise
            mask = (df_MLD_average['Station'] == station) & (df_MLD_average['Cruise'] == cruise)
            df_MLD_average.loc[mask, 'pHinsitu[Total]'] = avg_pH
            df_MLD_average.loc[mask, 'Chl_a[mg/m^3]'] = avg_chl

    # Keep only the columns you want
    df_MLD_average = df_MLD_average[['Station', 'Cruise', 'Lat [°N]', 'Lon [°E]', 'pHinsitu[Total]', 'Chl_a[mg/m^3]']]

    return df_MLD_average

def get_MLD_avg(df):
    """
    For each (Cruise, Station) group, compute the mean pHinsitu[Total] and Chl_a[mg/m^3] for rows where
    Sigma_theta[kg/m^3] is between sigma_surface (minimum in group) and sigma_surface+0.03.
    Also compute the mixed layer depth (MLD) as the difference between max and min depth in the mask.
    Returns a DataFrame with columns:
    ['Station', 'Cruise', 'Lat [°N]', 'Lon [°E]', 'pHinsitu[Total]', 'Chl_a[mg/m^3]', 'MLD']
    """
    results = []
    for (cruise, station), group in df.groupby(['Cruise', 'Station']):
        if 'Sigma_theta[kg/m^3]' not in group.columns or 'pHinsitu[Total]' not in group.columns:
            continue
        sigma_surface = group['Sigma_theta[kg/m^3]'].min()
        mask = (group['Sigma_theta[kg/m^3]'] >= sigma_surface) & (group['Sigma_theta[kg/m^3]'] <= sigma_surface + 0.03)
        mean_pH = group.loc[mask, 'pHinsitu[Total]'].mean()
        mean_chl = group.loc[mask, 'Chl_a[mg/m^3]'].mean() if 'Chl_a[mg/m^3]' in group.columns else np.nan
        lat = group['Lat [°N]'].iloc[0] if 'Lat [°N]' in group.columns else np.nan
        lon = group['Lon [°E]'].iloc[0] if 'Lon [°E]' in group.columns else np.nan
        if 'Depth[m]' in group.columns and mask.any():
            mld = group.loc[mask, 'Depth[m]'].max() - group.loc[mask, 'Depth[m]'].min()
        else:
            mld = np.nan
        results.append({
            'Station': station,
            'Cruise': cruise,
            'Lat [°N]': lat,
            'Lon [°E]': lon,
            'pHinsitu[Total]': mean_pH,
            'Chl_a[mg/m^3]': mean_chl,
            'MLD': mld
        })
    return pd.DataFrame(results)
# Define variable-specific percentile limits
def get_clim(df, color_column):
    if color_column == 'ChlorophyllA':
        lower, upper = np.percentile(df[color_column].dropna(), [5, 99])
    else:
        lower, upper = np.percentile(df[color_column].dropna(), [1, 99])

    step = max(round((upper - lower) / 100, 3), 0.001)

    return lower, upper, step

def filter_glider_assets(df, glider_overlay):
    """ Fliter gliderviz files based on selected gliders from assets menu. """
    if glider_overlay:
            # Extract numeric parts from the selected cruise strings (e.g., 'SN203' -> '203')
            selected_cruises = [s.replace('SN', '') for s in glider_overlay if 'SN' in s]
            # Apply filter for any of the selected cruises
            df_filtered = df[df["Cruise"].astype(str).str.contains('|'.join(selected_cruises))]
    else:
        df_filtered = df.iloc[0:0] 

    return df_filtered

def apply_common_scatter_styling(fig, colorbar_title="Station", colorbar_orientation='v'):
    """
    Apply common layout styling to a Plotly scatter figure.

    Parameters:
    ----------
    fig : plotly.graph_objects.Figure or plotly.express.Figure
        The figure to style.
    colorbar_title : str
        Title for the colorbar.
    colorbar_orientation : str
        'v' for vertical (default), 'h' for horizontal.
    """
    fig.update_yaxes(autorange="reversed")
    
    colorbar_config = dict(
        title=dict(text=colorbar_title),
        orientation=colorbar_orientation,
        len=0.8,
        thickness=25,
    )
    
    if colorbar_orientation == 'v':
        colorbar_config.update(x=1.02, y=0.5)
    else:  # horizontal
        colorbar_config.update(x=0.5, y=-0.2, xanchor='center', yanchor='top')

    fig.update_layout(coloraxis_colorbar=colorbar_config)
    return fig

def make_depth_scatter_plot(
    df, x, y="Depth[m]", title=None,
    color="Station", symbol="Cruise",
    labels=None,
    colorbar_title=None,
    colorbar_orientation='v'
):
    """
    Create a reusable scatter plot of a variable vs. depth with consistent styling.

    Parameters:
    ----------
    df : DataFrame
        Input data.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis (default: "Depth[m]").
    title : str or None
        Plot title. If None, a default is generated.
    color : str
        Column to use for color.
    symbol : str
        Column to use for symbol.
    labels : dict or None
        Custom axis labels.
    colorbar_title : str or None
        Title for the colorbar. Defaults to the `color` column name.
    colorbar_orientation : str
        'v' or 'h' for vertical or horizontal colorbar.

    Returns:
    -------
    fig : plotly.graph_objects.Figure
        Styled scatter plot.
    """
    if title is None:
        title = f"{x} vs. {y}"
    if labels is None:
        labels = {x: x, y: y, color: color, symbol: symbol}
    if colorbar_title is None:
        colorbar_title = color

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        symbol=symbol,
        labels=labels,
        title=title
    )

    # Reverse y-axis (for depth plots)
    fig.update_yaxes(autorange="reversed")

    # Add colorbar settings
    colorbar_config = dict(
        title=dict(text=colorbar_title),
        orientation=colorbar_orientation,
        len=0.8,
        thickness=25
    )

    if colorbar_orientation == 'v':
        colorbar_config.update(x=1.02, y=0.5)
    else:
        colorbar_config.update(x=0.5, y=-0.2, xanchor='center', yanchor='top')

    fig.update_layout(coloraxis_colorbar=colorbar_config)

    return fig

def make_depth_line_plot(
    df, x, y="Depth[m]", title=None,
    color="Datetime",
    labels=None,
    colorbar_title=None,
    colorbar_orientation='v'
):
    """
    Create a reusable line plot of a variable vs. depth with consistent styling.

    Parameters:
    ----------
    df : DataFrame
        Input data.
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis (default: "Depth[m]").
    title : str or None
        Plot title. If None, a default is generated.
    color : str
        Column to use for color.
    symbol : str
        Column to use for symbol.
    labels : dict or None
        Custom axis labels.
    colorbar_title : str or None
        Title for the colorbar. Defaults to the `color` column name.
    colorbar_orientation : str
        'v' or 'h' for vertical or horizontal colorbar.

    Returns:
    -------
    fig : plotly.graph_objects.Figure
        Styled line plot.
    """
    if title is None:
        title = f"{x} vs. {y}"
    if labels is None:
        labels = {x: x, y: y, color: color}
    if colorbar_title is None:
        colorbar_title = color
        # df = df.sort_values(by=[color, 'DIVEDIR', 'Depth[m]']) # add divedir
        # df = df.sort_values(by=[color, 'DIVEDIR' ,'Depth[m]']) # add divedir
        df = df.sort_values(by=['Station', 'DIVEDIR']) # This works, except connects the surface lines
        
        df['Datetime'] = df["Datetime"].dt.strftime("%m/%d %H:%M")
    # Get unique values and assign colors
    unique_vals = df[color].unique()
    num_colors = len(unique_vals)
    # viridis_colors = px.colors.sample_colorscale("Cividis", [i / (num_colors - 1) for i in range(num_colors)])
    if num_colors == 1:
        viridis_colors = px.colors.sample_colorscale("Cividis", 0.5)  # Just one mid-scale color

    else:
        viridis_colors = px.colors.sample_colorscale("Cividis", [i / (num_colors - 1) for i in range(num_colors)])
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        labels=labels,
        title=title,
        markers=True,
        color_discrete_sequence=viridis_colors,
        line_group=color # Each color is a different line
    )

    # Reverse y-axis (for depth plots)
    fig.update_yaxes(autorange="reversed")

    # Add colorbar settings
    colorbar_config = dict(
        title=dict(text=colorbar_title),
        orientation=colorbar_orientation,
        len=0.8,
        thickness=25
    )

    if colorbar_orientation == 'v':
        colorbar_config.update(x=1.02, y=0.5)
    else:
        colorbar_config.update(x=0.5, y=-0.2, xanchor='center', yanchor='top')

    fig.update_layout(coloraxis_colorbar=colorbar_config)

    # fig.update_layout(
    # legend=dict(
    #     x=0.01,
    #     y=0.99,
    #     xanchor='left',
    #     yanchor='top',
    #     bgcolor='rgba(255,255,255,0.6)',  # semi-transparent white
    #     bordercolor='black',
    #     borderwidth=1,
    #     traceorder='normal',
    #     font=dict(size=10),
    # )
    # )
    # fig.update_layout(
    #     legend=dict(
    #         orientation="h",       # horizontal layout
    #         yanchor="bottom",      # anchor the legend box from its bottom
    #         y=1.02,                # slightly above the plot area (1.0 is top edge)
    #         xanchor="left",
    #         x=0,
    #         bgcolor='rgba(255,255,255,0.7)',  # optional background
    #         bordercolor='black',
    #         borderwidth=1,
    #         font=dict(size=10)
    #     )
    # )

    return fig

def range_slider_marks(df, target_mark_count=10):
    """
    Generate RangeSlider marks at evenly spaced full-hour intervals,
    aligned to the nearest hour, based on a target number of marks.

    Parameters:
    ----------
    df : pandas.DataFrame
        Must contain 'Datetime' and 'unixTimestamp' columns.
    target_mark_count : int
        Approximate number of marks to generate.

    Returns:
    -------
    dict
        Dictionary of {unixTimestamp: formatted datetime string}
    """
    # Sort and get min/max
    df = df.sort_values("Datetime")
    t_min = df['Datetime'].min()
    t_max = df['Datetime'].max()

    if pd.isna(t_min) or pd.isna(t_max) or t_max <= t_min:
        return {}

    # Round t_min up to next full hour
    t_start = (t_min + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    # Total range in seconds
    total_seconds = (t_max - t_start).total_seconds()
    if total_seconds <= 0:
        return {}

    # Compute spacing interval (rounded to nearest hour step)
    interval_seconds = total_seconds // target_mark_count
    interval_hours = max(1, int(round(interval_seconds / 3600)))

    # Generate evenly spaced timestamps
    timestamps = pd.date_range(start=t_start, end=t_max, freq=f'{interval_hours}h')

    # Convert to Unix timestamp and format labels
    # marks = {int(ts.timestamp()): ts.strftime('%m/%d %H:%M') for ts in timestamps}
    marks = {
        int(ts.timestamp()): {
            'label': ts.strftime('%m/%d') + '\n' + ts.strftime('%H:%M'),
            'style': {'fontSize': '12px', 'whiteSpace': 'pre'}
        }
        for ts in timestamps
    }

    return marks

gs = GulfStreamLoader()
GulfStreamBounds = gs.load_data()
glider_ids = ['SN203', 'SN209', 'SN210', 'SN211','SN069']

# Initialize the app with a Bootstrap theme
external_stylesheets = cast(List[str | Dict[str, Any]], [dbc.themes.FLATLY])
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server # Required for Gunicorn


# loader = GliderDataLoader(filenames=['25420901RT.txt', '25520301RT.txt', '25706901RT.txt'])
loader = GliderDataLoader(filenames=['25706901RT.txt', '25720901RT.txt', '25821001RT.txt', '25820301RT.txt'])
# Load the most recent file (automatically done if no filename provided)
df_latest = loader.load_data()
map_loader = MapDataLoader()
df_map = map_loader.load_data()
station_min, station_max = df_latest["Station"].min(), df_latest["Station"].max()
date_min, date_max = df_latest["Date"].min(), df_latest["Date"].max() 
unix_min, unix_max = df_latest["unixTimestamp"].min(), df_latest["unixTimestamp"].max() 
unix_max_minus_12hrs = unix_max - 60*60*12
marks = range_slider_marks(df_latest, 20)
datetime_max = df_latest["Datetime"].max()
# Need function to replace these!
utc_str = datetime_max.strftime("%Y-%m-%d %H:%M:%S")
et_str = datetime_max.tz_localize("UTC").tz_convert("US/Eastern").strftime("%Y-%m-%d %H:%M:%S")
pt_str = datetime_max.tz_localize("UTC").tz_convert("US/Pacific").strftime("%Y-%m-%d %H:%M:%S")
update_str = f'Last Updated: {utc_str} UTC | {et_str} ET | {pt_str} PT'
app.layout = dbc.Container([
    # Top row - Header
    dbc.Row([
        dbc.Col([
            html.H2('NESSIE', className='text-info text-start',
                    style={'fontFamily': 'Segoe UI, sans-serif', 'marginBottom': '10px'}),
        #     html.P([
        #     "Data Visualization for the ",
        #     html.A("LOCNESS Project", href="https://subhaslab.whoi.edu/loc-ness/", target="_blank", className="text-info")
        # ]),
            html.P(
                id='update-text',
                className='text-muted text-start',
                style={'fontFamily': 'Times New Roman', 'marginBottom': '10px', 'fontSize': '15px', 'color': '#2c3e50'}
            ),
            html.P(
                'GliderID, Next Surface Time (ET), lat, lon, +/- distance [m]',
                className='text-muted text-start',
                style={'fontFamily': 'Times New Roman', 'marginBottom': '10px', 'fontSize': '16px', 'fontWeight': 'bold', 'color': '#34495e'}
            ),
            html.P(
                id='update-projection-text-069',
                className='text-muted text-start',
                style={'fontFamily': 'Times New Roman', 'marginBottom': '10px', 'fontSize': '15px', 'color': '#7f8c8d', 'fontStyle': 'bold'}
            ),
            html.P(
                id='update-projection-text-209',
                className='text-muted text-start',
                style={'fontFamily': 'Times New Roman', 'marginBottom': '10px', 'fontSize': '15px', 'color': '#7f8c8d', 'fontStyle': 'bold'}
            ),
            html.P(
                id='update-projection-text-210',
                className='text-muted text-start',
                style={'fontFamily': 'Times New Roman', 'marginBottom': '10px', 'fontSize': '15px', 'color': '#7f8c8d', 'fontStyle': 'bold'}
            ),
        ], width=12)
    ]),
    # Map row
    dbc.Row([
        dbc.Col([
            html.Div([
                # Map plot
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='map-plot',
                            style={'height': '70vh', 'width': '100%'}
                        ),
                        dcc.Interval(id="interval-refresh", interval=60*1000*5, n_intervals=0)  # every 5 minutes
                    ])
                ]),
                # Range slider
                html.Div([
                    dcc.RangeSlider(
                        id='RangeSlider',
                        updatemode='mouseup',  # don't let it update till mouse released
                        min=unix_min,
                        max=unix_max,
                        step=3600, # 1 hour
                        value=[unix_max_minus_12hrs, unix_max],
                        marks=marks,
                        allowCross=False,
                        # tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], style={
                    'padding': '10px 5px',
                    'marginTop': '10px',
                    'width': '100%',
                    'boxSizing': 'border-box'
                })
            ], style={'padding': '10px', 'backgroundColor': '#e3f2fd'})
        ], width=12)
    ]),
    # Bottom row - Settings and Parameters side by side
    dbc.Row([
        # Gliders column
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Gliders:"),
                        dcc.Checklist(
                            id='glider_overlay_checklist',
                            options=[{'label': s, 'value': s} for s in glider_ids],
                            value=[],
                            labelStyle={'display': 'block'}
                        )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], xs=12, sm=12, md=6, lg=4, xl=2),
        # Assets column
        # dbc.Col([
        #     html.Div([
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H4("Assets:"),
        #                 dcc.Checklist(
        #                     id='assets_checklist',
        #                     options=[
        #                         {'label': 'RV Connecticut', 'value': 'RV Connecticut'},
        #                         {'label': 'LRAUV', 'value': 'LRAUV'},
        #                         {'label': 'Drifter A', 'value': 'Drifter A'},
        #                         {'label': 'Drifter B', 'value': 'Drifter B'},
        #                         {'label': 'Drifter C', 'value': 'Drifter C'}
        #                     ],
        #                     value=[],
        #                     labelStyle={'display': 'block'}
        #                 )
        #             ])
        #         ])
        #     ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        # ], xs=12, md=2),
        # Map parameters column
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Map Color Variable"),
                        dcc.RadioItems(
                            id='Parameters',
                            options=[
                                {'label': 'Datetime', 'value': 'unixTimestamp'},
                                {'label': 'temperature [^oC]', 'value': 'temperature'},
                                {'label': 'salinity [psu]', 'value': 'salinity'},
                                {'label': 'pHin', 'value': 'pHin'},
                                {'label': 'rhodamine', 'value': 'rhodamine'},
                                {'label': 'MLD [m]', 'value': 'MLD'},
                            ],
                            value='unixTimestamp'
                        )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], xs=12, sm=12, md=6, lg=4, xl=2),
        # Map parameters column
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Layers"),
                        dcc.RadioItems(
                            id='Layers',
                            options=[
                                {'label': 'Mixed Layer Depth Avg', 'value': 'MLD'},
                                {'label': 'Surface', 'value': 'Surface'}
                            ],
                            value='MLD'
                        )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], xs=12, sm=12, md=6, lg=4, xl=2),
        # Cast Direction column
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Cast Direction"),
                        dcc.RadioItems(
                            id='CastDirection',
                            options=[
                                {'label': 'Up+Down', 'value': 'UpDown'},
                                {'label': 'Upcast', 'value': 'Up'},
                                {'label': 'Downcast', 'value': 'Down'},
                                {'label': 'Mean', 'value': 'Mean'}
                            ],
                            value='Up'
                        )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], xs=12, sm=12, md=6, lg=4, xl=2),
        # Map Options column
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Map Options"),
                        dcc.Checklist(
                                id='map_options',
                                options=[
                                    {'label': 'Drifters', 'value': 'drifters'},
                                    {'label': 'Overlay Gulf Stream', 'value': 'overlay'},
                                    {'label': 'Overlay Glider Grid', 'value': 'glider_grid'},
                                    {'label': 'Overlay Stellwagen Bank MPA', 'value': 'mpa'},
                                    {'label': 'Overlay gomofs tracks', 'value': 'gomofs'},
                                    {'label': 'Overlay doppio tracks', 'value': 'doppio'}
                                    ],
                                value=[],
                                labelStyle={'display': 'block'}
                            )
                    ])
                ])
            ], style={'padding': '1vh', 'margin-top': '10px','margin-bottom': '10px'})
        ], xs=12, sm=12, md=6, lg=4, xl=2),
    ]),
    # Third Row - Plotting
    # (Removed main plots row)
    # Tabs for lazy loading
    dcc.Tabs(id='lazy-tabs', value='tab-phin-rho', children=[
        dcc.Tab(label='pH & Rhodamine', value='tab-phin-rho', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='pHin-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='rho-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
            ])
        ]),
        dcc.Tab(label='Temperature and Salinity', value='tab-1', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='temp-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='salinity-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
            ])
        ]),
        dcc.Tab(label='Oxygen & Sigma Theta', value='tab-oxy-chl', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='doxy-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='chl-plot', style={'height': '70vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=6, lg=6, xl=6),
            ])
        ]),
        dcc.Tab(label='pH Diagnostics', value='tab-2', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vrs-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vrs-std-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='ik-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vk-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='vk-std-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='ib-plot', style={'height': '40vh', 'minHeight': '300px', 'width': '100%'})
                        ])
                    ])
                ], xs=12, sm=12, md=12, lg=4, xl=4),
            ]),
        ]),
        # --- Property Plot Tab ---
        dcc.Tab(label='Property Plot', value='tab-property', children=[
            dbc.Row([
                dbc.Col([
                    html.Label('Select X-axis:'),
                    dcc.Dropdown(id='property-x-dropdown'),
                ], xs=12, sm=12, md=3, lg=3, xl=3),
                dbc.Col([
                    html.Label('Select Y-axis:'),
                    dcc.Dropdown(id='property-y-dropdown'),
                ], xs=12, sm=12, md=3, lg=3, xl=3),
            ], className='mb-3'),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='property-plot', style={'height': '60vh', 'minHeight': '300px', 'width': '100%'})
                ], xs=12, sm=12, md=12, lg=12, xl=12)
            ])
        ]),
        # # --- Interpolation Map Tab ---
        # dcc.Tab(label='Interpolation Map', value='tab-interpolation', children=[
        #     dbc.Row([
        #         dbc.Col([
        #             html.Label('Select Parameter:'),
        #             dcc.Dropdown(id='interpolation-parameter-dropdown',options=[{'label': 'pHin', 'value': 'pHin'},
        #              {'label': 'rhodamine', 'value': 'rhodamine'}, {'label': 'temperature', 'value': 'temperature'},
        #               {'label': 'salinity', 'value': 'salinity'}], value='rhodamine'),
        #         ], xs=12, sm=12, md=3, lg=3, xl=3),
        #         dbc.Col([
        #             html.Label('Hours Back:'),
        #             dcc.Slider(id='interpolation-hours-back-slider', min=1, max=24, step=1, value=3),
        #         ], xs=12, sm=12, md=3, lg=3, xl=3),
        #         dbc.Col([
        #             html.Label('Select Platform:'),
        #             dcc.Dropdown(id='interpolation-platform-dropdown',options=[{'label': 'Glider', 'value': 'Glider'},
        #              {'label': 'Ship', 'value': 'Ship'}, {'label': 'LRAUV', 'value': 'LRAUV'}], value=['Glider','Ship','LRAUV'],multi=True),
        #         ], xs=12, sm=12, md=3, lg=3, xl=3),
        #     ], className='mb-3'),
        #     dbc.Row([
        #         dbc.Col([
        #             dcc.Graph(id='interpolation-map', figure=go.Figure(), style={'height': '60vh', 'minHeight': '300px', 'width': '100%'})
        #         ], xs=12, sm=12, md=12, lg=12, xl=12)
        #     ])
        # ]),
    ]),
], fluid=True, className='dashboard-container')

# --- Combined callback for map and scatter plots ---
@app.callback(
    [Output('map-plot', 'figure'),
     Output('pHin-plot', 'figure'),
     Output('doxy-plot', 'figure'),
     Output('temp-plot', 'figure'),
     Output('salinity-plot', 'figure'),
     Output('chl-plot', 'figure'),
     Output('rho-plot', 'figure'),
     Output('vrs-plot', 'figure'),
     Output('vrs-std-plot', 'figure'),
     Output('vk-plot', 'figure'),
     Output('vk-std-plot', 'figure'),
     Output('ik-plot', 'figure'),
     Output('ib-plot', 'figure'),
     Output('property-plot', 'figure'),  # New output for property plot
     Output('property-x-dropdown', 'options'), # New output for x dropdown options
     Output('property-y-dropdown', 'options')  # New output for y dropdown options
    ],
    [Input('interval-refresh', 'n_intervals'),
     Input('Parameters', 'value'),
     Input('map_options', 'value'),
     Input('glider_overlay_checklist', 'value'),
     Input('lazy-tabs', 'value'),
     Input('RangeSlider', 'value'),
     Input('Layers', 'value'),
     Input('CastDirection', 'value'),
     Input('property-x-dropdown', 'value'), # New input for x axis
     Input('property-y-dropdown', 'value')  # New input for y axis
    ]
)
def update_all_figs(n, selected_parameter, map_options, glider_overlay, selected_tab, range_value, selected_layer, selected_cast_direction, property_x, property_y):
    # Load glider and filter by selected gliders
    df_latest = loader.load_data()
    df_latest = filter_glider_assets(df_latest, glider_overlay)
    # Load map data
    df_map = map_loader.load_data()
    # load glider grid
    glider_grid_loader = GliderGridDataLoader()
    df_glider_grid = glider_grid_loader.load_data()
    # load mpa data
    mpa_loader = MPADataLoader()
    df_mpa = mpa_loader.load_data()
    # load gomofs
    # Filter by date range
    if range_value[0] == range_value[1]:
        df_latest_filter = df_latest
        df_map_filtered = df_map
    else:
        df_latest_filter = df_latest[
            (df_latest['unixTimestamp'] >= range_value[0]) &
            (df_latest['unixTimestamp'] <= range_value[1])
        ]
        df_map_filtered = df_map[
            (df_map['unixTimestamp'] >= range_value[0]) &
            (df_map['unixTimestamp'] <= range_value[1])
        ]
    df_drifters = df_map_filtered[df_map_filtered['Platform'] == "Drifter"] # Drifter data only filtered by time
    # Preserve all WPT rows
    wpt_rows = df_map_filtered[df_map_filtered['Layer'] == 'WPT']

    # Filter by layers (only for non-WPT rows)
    non_wpt_rows = df_map_filtered[df_map_filtered['Layer'] != 'WPT']
    if selected_layer == 'MLD':
        non_wpt_rows = non_wpt_rows[non_wpt_rows['Layer'] == 'MLD']
    elif selected_layer == 'Surface':
        non_wpt_rows = non_wpt_rows[non_wpt_rows['Layer'] == 'Surface']

    # Recombine WPT and filtered rows
    df_map_filtered = pd.concat([non_wpt_rows, wpt_rows], ignore_index=True)
    # Filter by cast direction
    if selected_cast_direction == 'Mean':
        df_map_filtered = df_map_filtered[(df_map_filtered['CastDirection'] == 'Mean') | (df_map_filtered['CastDirection'] == 'Constant')]
    elif selected_cast_direction == 'UpDown':
        df_map_filtered = df_map_filtered[(df_map_filtered['CastDirection'] == 'Up') | (df_map_filtered['CastDirection'] == 'Down') | (df_map_filtered['CastDirection'] == 'Constant')]
    elif selected_cast_direction == 'Up':
        df_map_filtered = df_map_filtered[(df_map_filtered['CastDirection'] == 'Up') | (df_map_filtered['CastDirection'] == 'Constant')]
        df_latest_filter = df_latest_filter[df_latest_filter["DIVEDIR"] == 1]
    elif selected_cast_direction == 'Down':
        df_map_filtered = df_map_filtered[(df_map_filtered['CastDirection'] == 'Down') | (df_map_filtered['CastDirection'] == 'Constant')]
        df_latest_filter = df_latest_filter[df_latest_filter["DIVEDIR"] == -1]
    # Dataframes for map plot
    df_ship = df_map_filtered[df_map_filtered['Cruise'] == "RV Connecticut"]
    df_LRAUV = df_map_filtered[df_map_filtered['Platform'] == "LRAUV"]
    df_SN209 = df_map_filtered[(df_map_filtered['Cruise'] == "25720901") & (df_map_filtered['Layer'] != 'WPT')]
    df_SN210 = df_map_filtered[(df_map_filtered['Cruise'] == "25821001") & (df_map_filtered['Layer'] != 'WPT')]
    df_SN069 = df_map_filtered[(df_map_filtered['Cruise'] == "25706901") & (df_map_filtered['Layer'] != 'WPT')]
    df_SN209_nxt = df_map_filtered[(df_map_filtered['Cruise'] == "25720901") & (df_map_filtered['Layer'] == 'WPT')]
    df_SN210_nxt = df_map_filtered[(df_map_filtered['Cruise'] == "25821001") & (df_map_filtered['Layer'] == 'WPT')]
    df_SN069_nxt = df_map_filtered[(df_map_filtered['Cruise'] == "25706901") & (df_map_filtered['Layer'] == 'WPT')]
    #  Weird issue with wpt when range slider is adjusted
    # Handle empty DataFrame case
    is_map_df = isinstance(df_map_filtered, pd.DataFrame)
    is_latest_df = isinstance(df_latest_filter, pd.DataFrame)
    if not is_map_df or df_map_filtered.empty:
        blank_fig = go.Figure()
        return (
            blank_fig, blank_fig, blank_fig,
            blank_fig, blank_fig, blank_fig,
            blank_fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            go.Figure(), [], []
        )
    # Set map center
    if len(df_ship) > 0:
        last_ship_lat = np.array(df_ship['lat'])[-1]
        last_ship_lon = np.array(df_ship['lon'])[-1]
    else:
        last_ship_lat = np.array(df_map_filtered['lat'])[-1]
        last_ship_lon = np.array(df_map_filtered['lon'])[-1]

    if len(df_LRAUV) > 0:
        last_LRAUV_lat = np.array(df_LRAUV['lat'])[-1]
        last_LRAUV_lon = np.array(df_LRAUV['lon'])[-1]
    else:
        last_LRAUV_lat = []
        last_LRAUV_lon = []

    # Last Glider Location SN209
    if len(df_SN209) > 0:
        last_glider_lat_SN209 = np.array(df_SN209['lat'])[-1]
        last_glider_lon_SN209 = np.array(df_SN209['lon'])[-1]
        next_glider_lat_SN209 = np.array(df_SN209_nxt['lat'])[-1]
        next_glider_lon_SN209 = np.array(df_SN209_nxt['lon'])[-1]
    else:
        last_glider_lat_SN209 = []
        last_glider_lon_SN209 = []
        next_glider_lat_SN209 = []
        next_glider_lon_SN209 = []
    # Last Glider Location SN210
    if len(df_SN210) > 0:
        last_glider_lat_SN210 = np.array(df_SN210['lat'])[-1]
        last_glider_lon_SN210 = np.array(df_SN210['lon'])[-1]
        next_glider_lat_SN210 = np.array(df_SN210_nxt['lat'])[-1]
        next_glider_lon_SN210 = np.array(df_SN210_nxt['lon'])[-1]
    else:
        last_glider_lat_SN210 = []
        last_glider_lon_SN210 = []
        next_glider_lat_SN210 = []
        next_glider_lon_SN210 = []
    # Last Glider Location SN069
    if len(df_SN069) > 0:
        last_glider_lat_SN069 = np.array(df_SN069['lat'])[-1]
        last_glider_lon_SN069 = np.array(df_SN069['lon'])[-1]
        next_glider_lat_SN069 = np.array(df_SN069_nxt['lat'])[-1]
        next_glider_lon_SN069 = np.array(df_SN069_nxt['lon'])[-1]
    else:
        last_glider_lat_SN069 = []
        last_glider_lon_SN069 = []
        next_glider_lat_SN069 = []
        next_glider_lon_SN069 = []
    map_fig = go.Figure()
    # Set hard color limits for pHin and rhodamine
    if selected_parameter == 'pHin' and selected_layer == 'Surface':
        cmin, cmax = 8, 8.2
        cscale = 'bluered'
    elif selected_parameter == 'pHin' and selected_layer == 'MLD':
        cmin, cmax = 8, 8.2
        cscale = 'bluered'
    elif selected_parameter == 'rhodamine':
        # Default range for most traces
        cmin, cmax = 0, 10
        cscale = 'Reds'
    else:
        cmin, cmax = None, None
        cscale = 'Cividis'
    # Only do this if coloring by unixTimestamp
    if selected_parameter == 'unixTimestamp' and len(df_SN069) > 0:
        # Get unique unixTimestamps and corresponding datetimes
        unix_vals = df_SN069['unixTimestamp'].values
        datetimes = df_SN069['Datetime'].dt.strftime('%m/%d %H:%M').values

        # Choose 5 evenly spaced indices
        n_ticks = 5
        if len(unix_vals) > n_ticks:
            idxs = np.linspace(0, len(unix_vals) - 1, n_ticks, dtype=int)
            tickvals = unix_vals[idxs]
            ticktext = datetimes[idxs]
        else:
            tickvals = unix_vals
            ticktext = datetimes
    else:
        tickvals = None
        ticktext = None
   
    # Set hovertext based on selected parameter
    if len(df_ship) > 0:
        ship_hovertext = df_ship['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_ship[selected_parameter]
        ship_hovertext_last = np.array(ship_hovertext)[-1]
        map_fig.add_trace(go.Scattermap(
                lat=df_ship['lat'],
                lon=df_ship['lon'],
                mode='lines+markers',
                name='RV Connecticut',
                hovertext=ship_hovertext,
                marker=dict(
                    size=10,
                    color=df_ship[selected_parameter],
                    colorscale=cscale,
                    showscale=True,
                    colorbar=dict(
                    len=0.6,              # length of colorbar
                    thickness=15,         # width (since vertical)
                    tickvals=tickvals,
                    ticktext=ticktext,
                    x=0.5,               # near the right edge of the plot, 0.98 v
                    y=0.05,                # center vertically, 0.5 v
                    xanchor='center',      # anchor the right side of the bar to right v
                    yanchor='bottom',     # anchor the middle vertically, 
                    orientation='h'     # optional, vertical is default
                    ),
                    # colorbar=dict(len=0.6,tickvals=tickvals,ticktext=ticktext),
                    cmin=cmin,
                    cmax=cmax,
                ),
            ))
        map_fig.add_trace(go.Scattermap(
            lat=[last_ship_lat],
            lon=[last_ship_lon],
            mode='markers',
            name='RV Connecticut Last Location',
            hovertext=ship_hovertext_last,
            marker=dict(
                size=15,
                symbol='ferry',
                showscale=False,
            ),
            showlegend=False
        ))

    # Set hovertext based on selected parameter
    if len(df_LRAUV) > 0:
        LRAUV_hovertext = df_LRAUV['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_LRAUV[selected_parameter]
        LRAUV_hovertext_last = np.array(LRAUV_hovertext)[-1]
        map_fig.add_trace(go.Scattermap(
                lat=df_LRAUV['lat'],
                lon=df_LRAUV['lon'],
                mode='markers',
                name='LRAUV',
                hovertext=LRAUV_hovertext,
                marker=dict(
                    size=10,
                    color=df_LRAUV[selected_parameter],
                    colorscale=cscale,
                    showscale=False,
                    cmin=cmin,
                    cmax=cmax,
                ),
            ))
        map_fig.add_trace(go.Scattermap(
            lat=[last_LRAUV_lat],
            lon=[last_LRAUV_lon],
            mode='markers',
            name='LRAUV Last Location',
            hovertext=LRAUV_hovertext_last,
            marker=dict(
                size=10,
                symbol='heliport',
                showscale=False,
            ),
            showlegend=False
        ))

    # Set hovertext for SN209 based on selected parameter
    if len(df_SN209) > 0:
        sn209_hovertext = df_SN209['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_SN209[selected_parameter]
        sn209_hovertext_proj = df_SN209_nxt['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_SN209_nxt[selected_parameter]
        sn209_hovertext_last = np.array(sn209_hovertext)[-1]
        sn209_hovertext_nxt = np.array(sn209_hovertext_proj)[-1]
        map_fig.add_trace(go.Scattermap(
        lat=df_SN209['lat'],
        lon=df_SN209['lon'],
        mode='markers',
        name='SN209',
        hovertext=sn209_hovertext,
        marker=dict(
            size=10, 
            color=df_SN209[selected_parameter],
            colorscale=cscale,
            showscale=False,
            cmin=cmin,
            cmax=cmax,
        ),
        ))
        map_fig.add_trace(go.Scattermap(
        lat=[next_glider_lat_SN209],
        lon=[next_glider_lon_SN209],
        mode='markers',
        name='SN209 Next Location',
        hovertext=sn209_hovertext_nxt,
        marker=dict(
            size=15,
            symbol='swimming',
            showscale=False,
        ),
        showlegend=False
        ))
        map_fig.add_trace(go.Scattermap(
            lat=[last_glider_lat_SN209],
            lon=[last_glider_lon_SN209],
            mode='markers',
            name='SN209 Last Location',
            hovertext=sn209_hovertext_last,
            marker=dict(
                size=10,
                symbol='airport',
                showscale=False,
            ),
            showlegend=False
        ))
    # Set hovertext for SN210 based on selected parameter
    if len(df_SN210) > 0:
        sn210_hovertext = df_SN210['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_SN210[selected_parameter]
        sn210_hovertext_proj = df_SN210_nxt['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_SN210_nxt[selected_parameter]
        sn210_hovertext_last = np.array(sn210_hovertext)[-1]
        sn210_hovertext_nxt = np.array(sn210_hovertext_proj)[-1]
        map_fig.add_trace(go.Scattermap(
        lat=df_SN210['lat'],
        lon=df_SN210['lon'],
        mode='markers',
        name='SN210',
        hovertext=sn210_hovertext,
        marker=dict(
            size=10, 
            color=df_SN210[selected_parameter],
            colorscale=cscale,
            showscale=False,
            cmin=cmin,
            cmax=cmax,
        ),
        ))
        map_fig.add_trace(go.Scattermap(
        lat=[next_glider_lat_SN210],
        lon=[next_glider_lon_SN210],
        mode='markers',
        name='SN210 Next Location',
        hovertext=sn210_hovertext_nxt,
        marker=dict(
            size=15,
            symbol='swimming',
            showscale=False,
        ),
        showlegend=False
        ))
        map_fig.add_trace(go.Scattermap(
            lat=[last_glider_lat_SN210],
            lon=[last_glider_lon_SN210],
            mode='markers',
            name='SN210 Last Location',
            hovertext=sn210_hovertext_last,
            marker=dict(
                size=10,
                symbol='airport',
                showscale=False,
            ),
            showlegend=False
        ))
    # Set hovertext for SN069 based on selected parameter
    if len(df_SN069) > 0:
        sn069_hovertext = df_SN069['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_SN069[selected_parameter]
        sn069_hovertext_proj = df_SN069_nxt['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S') if selected_parameter == 'unixTimestamp' else df_SN069_nxt[selected_parameter]
        sn069_hovertext_last = np.array(sn069_hovertext)[-1]
        sn069_hovertext_nxt = np.array(sn069_hovertext_proj)[-1]
        # For rhodamine, SN069 uses different range (0-15) while others use (0-100)
        if selected_parameter == 'rhodamine':
            sn069_cmin, sn069_cmax = 0, 15
        else:
            sn069_cmin, sn069_cmax = cmin, cmax
            
        map_fig.add_trace(go.Scattermap(
            lat=df_SN069['lat'],
            lon=df_SN069['lon'],
            mode='markers',
            name='SN069',
            hovertext=sn069_hovertext,
            marker=dict(
                size=10, 
                color=df_SN069[selected_parameter],
                colorscale=cscale,  
                showscale=False,
                cmin=sn069_cmin,
                cmax=sn069_cmax,
            ),
        ))      
        map_fig.add_trace(go.Scattermap(
            lat=[next_glider_lat_SN069],
            lon=[next_glider_lon_SN069],
            mode='markers',
            name='SN069 Next Location',
            hovertext=sn069_hovertext_nxt,
            marker=dict(
                size=15,
                symbol='swimming',
                showscale=False,
            ),
            showlegend=False
        ))
        map_fig.add_trace(go.Scattermap(
            lat=[last_glider_lat_SN069],
            lon=[last_glider_lon_SN069],
            mode='markers',
            name='SN069 Last Location',
            hovertext=sn069_hovertext_last,
            marker=dict(
                size=10,
                symbol='airport',
                showscale=False,
            ),
            showlegend=False
        ))
    # Drifters
    if len(df_drifters) > 0 and 'drifters' in map_options and selected_parameter == 'unixTimestamp':
        for cruise in df_drifters['Cruise'].unique():
            df_cruise = df_drifters[df_drifters['Cruise'] == cruise]
            drifter_hovertext = df_cruise['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            map_fig.add_trace(go.Scattermap(
                    lat=df_cruise['lat'],  # Use df_cruise instead of df_drifters
                    lon=df_cruise['lon'],  # Use df_cruise instead of df_drifters
                    mode='markers',
                    name=f'SPOT{cruise[-2:]}',
                    hovertext=drifter_hovertext,
                    marker=dict(
                        size=8,
                        color=df_cruise['unixTimestamp'],  # Use df_cruise instead of df_drifters
                        colorscale=cscale,
                        showscale=False,
                        cmin=cmin,
                        cmax=cmax,
                    ),
                    text=df_cruise['Cruise'],  # Use df_cruise instead of df_drifters
                    textposition = "bottom right",
                ))
        
    if 'overlay' in map_options:
        map_fig.add_trace(go.Scattermap(
        lat=GulfStreamBounds['Lat'],
        lon=GulfStreamBounds['Lon'],
        mode='lines',
        name='Gulf Stream',
        marker=dict(size=6, color='deepskyblue'),
        line=dict(width=2, color='deepskyblue'),
        ))

    if 'glider_grid' in map_options:
        map_fig.add_trace(go.Scattermap(
            lat=df_glider_grid['Lat'],
            lon=df_glider_grid['Lon'],
            mode='markers',
            name='Glider Grid',
            marker=dict(size=6, color='black', opacity=0.5),
            text = df_glider_grid['Grid_ID'],
            textposition = "bottom right",
        ))

    if 'mpa' in map_options:
        map_fig.add_trace(go.Scattermap(
            lat=df_mpa['Lat'],
            lon=df_mpa['Lon'],
            mode='lines',
            name='Stellwagen Bank MPA',
            line=dict(width=4, color='red'),
            text = 'Stellwagen Bank MPA',
            textposition = "bottom right",
        ))
    if 'gomofs' in map_options:
        gomofs_loader = gomofsdataloader()
        df_gomofs = gomofs_loader.load_data()
        gomofshovertext = df_gomofs['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        map_fig.add_trace(go.Scattermap(
            lat=df_gomofs['lat'],
            lon=df_gomofs['lon'],
            mode='lines+markers',
            name='gomofs tracks',
            line=dict(width=2, color='red'),
            hovertext = gomofshovertext,
            textposition = "bottom right",
        ))
    if 'doppio' in map_options:
        doppio_loader = doppiodataloader()
        df_doppio = doppio_loader.load_data()
        doppioshovertext = df_doppio['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        map_fig.add_trace(go.Scattermap(
            lat=df_doppio['lat'],
            lon=df_doppio['lon'],
            mode='lines+markers',
            name='doppio tracks',
            line=dict(width=2, color='orange'),
            hovertext = doppioshovertext,
            textposition = "bottom right",
        ))

    # map_fig.update_layout(map = {'zoom': 8, 'style': 'satellite', 'center': {'lat': last_ship_lat, 'lon': last_ship_lon}})
    # map_fig.update_traces(marker_symbol=1, selector=dict(name='SPOT01'))

    map_fig.update_layout(
        map_style="white-bg",
        map_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                # "sourceattribution": "Esri, Garmin, GEBCO, NOAA NGDC, and other contributors",
                "source": [
                    "https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
                ]
            },
        ],
    )   
    map_fig.update_layout(
        map=dict(
            center={'lat': last_ship_lat, 'lon': last_ship_lon},
            zoom=8,
        )
    )
    map_fig.update_layout(margin={"r":0,"t":20,"l":0,"b":0})
    map_fig.update_layout(
    legend=dict(
        x=0.99,              # near the far right
        y=0.99,              # near the top
        xanchor='right',     # anchor the right side of the box at x
        yanchor='top',       # anchor the top of the box at y
        bgcolor='rgba(255,255,255,0.5)',  # semi-transparent background
        bordercolor='black',
        borderwidth=1
    )
)

    # For scatter plots: full filtered DataFrame
    if len(df_latest_filter) == 0:
        scatter_fig_pHin = go.Figure()
        scatter_fig_doxy = go.Figure()
        scatter_fig_temp = go.Figure()
        scatter_fig_salinity = go.Figure()
        scatter_fig_chl = go.Figure()
        scatter_fig_rho = go.Figure()
        scatter_fig_vrs = go.Figure()
        scatter_fig_vrs_std = go.Figure()
        scatter_fig_vk = go.Figure()
        scatter_fig_vk_std = go.Figure()
        scatter_fig_ik = go.Figure()
        scatter_fig_ib = go.Figure()
    else:
        scatter_fig_pHin = make_depth_line_plot(
            df_latest_filter,
            x="pHinsitu[Total]",
            title="pHinsitu[Total] vs. Depth"
        )
        scatter_fig_doxy = make_depth_line_plot(
            df_latest_filter,
            x="Oxygen[µmol/kg]",
            title="Oxygen[µmol/kg] vs. Depth"
        )
        scatter_fig_temp = make_depth_line_plot(
            df_latest_filter,
            x="Temperature[°C]",
            title="Temperature[°C] vs. Depth"
        )
        scatter_fig_salinity = make_depth_line_plot(
            df_latest_filter,
            x="Salinity[pss]",
            title="Salinity[pss] vs. Depth"
        )
        # Changed to Sigma!!!!!!!!!
        scatter_fig_chl = make_depth_line_plot(
            df_latest_filter,
            x="Sigma_theta[kg/m^3]",
            title="Sigma_theta[kg/m^3] vs. Depth"
        )
        scatter_fig_rho = make_depth_line_plot(
            df_latest_filter,
            x="RHODAMINE[ppb]",
            title="RHODAMINE[ppb] vs. Depth"
        )
        scatter_fig_vrs = make_depth_line_plot(
            df_latest_filter,
            x="VRS[Volts]",
            title="VRS[Volts] vs. Depth"
        )
        scatter_fig_vrs_std = make_depth_line_plot(
            df_latest_filter,
            x="VRS_STD[Volts]",
            title="VRS_STD[Volts] vs. Depth"
        )
        scatter_fig_vk = make_depth_line_plot(
            df_latest_filter,
            x="VK[Volts]",
            title="VK[Volts] vs. Depth"
        )
        scatter_fig_vk_std = make_depth_line_plot(
            df_latest_filter,
            x="VK_STD[Volts]",
            title="VK_STD[Volts] vs. Depth"
        )
        scatter_fig_ik = make_depth_line_plot(
            df_latest_filter,
            x="IK[nA]",
            title="IK[nA] vs. Depth"
        )
        scatter_fig_ib = make_depth_line_plot(
            df_latest_filter,
            x="Ib[nA]",
            title="Ib[nA] vs. Depth"
        )
    # Property plot dropdown options
    if len(df_latest_filter) > 0:
        dropdown_options = [
            {'label': col, 'value': col}
            for col in df_latest_filter.columns if 'QF' not in col
        ]
        # Get unique unixTimestamps and corresponding datetimes
        unix_vals = df_latest_filter['unixTimestamp'].values
        datetimes = df_latest_filter['Datetime'].dt.strftime('%m/%d %H:%M').values

        # Choose 10 evenly spaced indices
        n_ticks = 5
        if len(unix_vals) > n_ticks:
            idxs = np.linspace(0, len(unix_vals) - 1, n_ticks, dtype=int)
            tickvals = unix_vals[idxs]
            ticktext = datetimes[idxs]
        else:
            tickvals = unix_vals
            ticktext = datetimes
    else:
        dropdown_options = []
    # Property plot figure
    if selected_tab == 'tab-property' and property_x and property_y and len(df_latest_filter) > 0:
        fig_property = px.scatter(
            df_latest_filter, x=property_x, y=property_y,
            
            # colorbar=dict(len=1,tickvals=tickvals,ticktext=ticktext),
            title=f'{property_x} vs. {property_y}',
            template='plotly_white',
            color='unixTimestamp' if 'unixTimestamp' in df_latest_filter.columns else None,
        )
        if 'Depth[m]' in [property_x, property_y]:
            fig_property.update_yaxes(autorange="reversed")
        fig_property.update_traces(marker=dict(size=6))
        if 'unixTimestamp' in df_latest_filter.columns:
            fig_property.update_layout(
                coloraxis_colorbar=dict(
                    len=1,
                    tickvals=tickvals,
                    ticktext=ticktext,
                    title='Datetime'
                )
            )
    else:
        fig_property = go.Figure()
    # Always update map, pHin, doxy
    # Tab 1: update temp, salinity; Tab 2: update vrs, vrs_std, vk, vk_std, ik, ib; Tab-phin-rho: update pHin and Rho; Tab-oxy-chl: update O2 and Chl; Tab-property: update property plot
    if selected_tab == 'tab-phin-rho':
        return (
            map_fig, scatter_fig_pHin, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            scatter_fig_rho,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            go.Figure(), dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-1':
        return (
            map_fig, dash.no_update, dash.no_update,
            scatter_fig_temp, scatter_fig_salinity, dash.no_update,
            dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            go.Figure(), dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-oxy-chl':
        return (
            map_fig, dash.no_update, scatter_fig_doxy,
            dash.no_update, dash.no_update, scatter_fig_chl,
            dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            go.Figure(), dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-2':
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update,
            scatter_fig_vrs, scatter_fig_vrs_std, scatter_fig_vk, scatter_fig_vk_std, scatter_fig_ik, scatter_fig_ib,
            go.Figure(), dropdown_options, dropdown_options
        )
    elif selected_tab == 'tab-property':
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            fig_property, dropdown_options, dropdown_options
        )
    else:
        return (
            map_fig, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update,
            dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            go.Figure(), dropdown_options, dropdown_options
        )

# @app.callback(
#     Output('interpolation-map', 'figure'),
#     [Input('interpolation-parameter-dropdown', 'value'),
#      Input('interpolation-hours-back-slider', 'value'),
#      Input('interpolation-platform-dropdown', 'value')]
# )
# def update_interpolation(parameter, hours_back, platforms):
#     if not parameter or not platforms:
#         return go.Figure()
    
#     try:
#         # Get your filtered dataframe
#         df_map = map_loader.load_data()
        
#         if df_map.empty:
#             print("No data available for interpolation")
#             return go.Figure()
        
#         # print(f"Creating interpolation for {parameter} with {hours_back} hours back, platforms: {platforms}, layer: {layer}, method: {method}")
        
#         # Create interpolation
#         fig, metadata = create_spatial_interpolation(
#             df=df_map,
#             parameter=parameter,
#             hours_back=hours_back,
#             platform_filter=platforms,
#             layer_filter='MLD',
#             grid_resolution=80,
#             method='linear',
#             nan_filter_parameters=[parameter]
#         )
        
#         if fig is not None:
#             # print(f"Interpolation successful: {metadata}")
#             return fig
#         else:
#             # print("Interpolation failed - no figure returned")
#             return go.Figure()
            
    # except Exception as e:
    #     # print(f"Error in interpolation callback: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return go.Figure()


@app.callback(
    [
        Output('RangeSlider', 'min'),
        Output('RangeSlider', 'max'),
        Output('RangeSlider', 'value'),
        Output('RangeSlider', 'marks'),
        Output('update-text', 'children'),
        Output('update-projection-text-069', 'children'),
        Output('update-projection-text-209', 'children'),
        Output('update-projection-text-210', 'children')
    ],
    [
        Input('glider_overlay_checklist', 'value'),
        Input('interval-refresh', 'n_intervals'),
    ]
)
def update_range_slider(glider_overlay, n):
    try:
        df_map = map_loader.load_data()
        if df_map.empty:
            # Return safe defaults if no data - must return 8 values to match outputs
            return 0, 1, [0, 1], {0: "No data"}, "No data available", "No data", "No data", "No data"
        
        unix_min = df_map["unixTimestamp"].min()
        unix_max = df_map["unixTimestamp"].max()
        unix_max_minus_12hrs = unix_max - 60*60*12
        marks = range_slider_marks(df_map, 10)

        # Filter out WPT rows for datetime calculation
        non_wpt_data = df_map[df_map['Layer'] != 'WPT']
        if non_wpt_data.empty:
            return unix_min, unix_max, [unix_max_minus_12hrs, unix_max], marks, "No non-WPT data", "No data", "No data", "No data"
        
        datetime_max = non_wpt_data["Datetime"].max()
        utc_str = datetime_max.strftime("%Y-%m-%d %H:%M:%S")
        et_str = datetime_max.tz_localize("UTC").tz_convert("US/Eastern").strftime("%Y-%m-%d %H:%M:%S")
        pt_str = datetime_max.tz_localize("UTC").tz_convert("US/Pacific").strftime("%Y-%m-%d %H:%M:%S")
        update_str = f'Last Updated: {utc_str} UTC | {et_str} ET | {pt_str} PT'
        
        # Safely get projection data for each glider
        def get_glider_projection(df, cruise_id):
            try:
                glider_data = df[(df['Layer'] == 'WPT') & (df['Cruise'] == cruise_id)]
                if glider_data.empty:
                    return "No projection data"
                glider_row = glider_data.iloc[-1]
                dt_utc = glider_row["Datetime"]
                dt_ET = dt_utc.tz_localize("UTC").tz_convert("US/Eastern").strftime("%Y-%m-%d %H:%M:%S")
                return f'SN{cruise_id[-2:]}, {dt_ET} +/- 5 min, {glider_row["lat"]:.4f}, {glider_row["lon"]:.4f}, +/- 500m'
            except Exception as e:
                return f"Error: {str(e)}"
        
        update_projection_str_069 = get_glider_projection(df_map, '25706901')
        update_projection_str_209 = get_glider_projection(df_map, '25720901')
        update_projection_str_210 = get_glider_projection(df_map, '25821001')

        return unix_min, unix_max, [unix_max_minus_12hrs, unix_max], marks, update_str, update_projection_str_069, update_projection_str_209, update_projection_str_210
    
    except Exception as e:
        # Return safe defaults on any error - must return 8 values to match outputs
        print(f"Error in update_range_slider: {e}")
        return 0, 1, [0, 1], {0: "Error"}, f"Error: {str(e)}", "Error", "Error", "Error"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Render dynamically assigns a port
    app.run(host="0.0.0.0", port=str(port), debug=True)