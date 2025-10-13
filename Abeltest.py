import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, html, dcc, Input, Output
import datetime as dt

# ------------------------------------------------------------
# 1. Load and clean sensor location data
# ------------------------------------------------------------
def load_sensor_locations(path="sensor-location.xlsx - Sheet1.csv"):
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Objectummer": "sensor_id",
            "Locatienaam": "location_name",
            "Lat/Long": "lat_long",
            "Breedte": "width_m",
            "Effectieve  breedte": "effective_width_m",
        }
    )
    lat_lon_split = df["lat_long"].str.split(",", expand=True)
    df["latitude"] = lat_lon_split[0].astype(float)
    df["longitude"] = lat_lon_split[1].astype(float)

    # normalize commas in width columns
    for col in ["width_m", "effective_width_m"]:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)

    df = df.drop(columns=["lat_long"])
    return df

# ------------------------------------------------------------
# 2. Load visitor data (confidential dataset)
# ------------------------------------------------------------
def load_visitor_data(path="sensordata_SAIL2025.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df

# ------------------------------------------------------------
# 3. Prepare data
# ------------------------------------------------------------
loc_df = load_sensor_locations()
vis_df = load_visitor_data()

# Extract sensor columns (ignore time features)
sensor_cols = [c for c in vis_df.columns if "_" in c]

start_time = pd.Timestamp("2025-08-20 00:00:00+02:00")
end_time = vis_df.index.max()

# ------------------------------------------------------------
# 4. Build Dash app
# ------------------------------------------------------------
app = Dash(__name__)

app.layout = html.Div(
    style={"display": "flex", "flexDirection": "row", "height": "100vh"},
    children=[
        # LEFT COLUMN - Weather Placeholder
        html.Div(
            style={"flex": "1", "padding": "20px", "background": "#f0f4f7"},
            children=[
                html.H2("Weather Forecast ☁️", style={"textAlign": "center"}),
                html.Div("Coming soon...", style={"textAlign": "center", "fontSize": "18px"}),
            ],
        ),

        # MIDDLE COLUMN - Map
        html.Div(
            style={"flex": "2", "padding": "10px"},
            children=[
                html.H2("Sensor Locations - Heatmap 🌍", style={"textAlign": "center"}),
                dcc.Graph(id="heatmap"),
                dcc.Slider(
                    min=0,
                    max=len(vis_df) - 1,
                    step=1,
                    value=0,
                    id="time-slider",
                    marks=None,
                    tooltip={"always_visible": True},
                    updatemode="drag",
                ),
                html.Div(id="time-display", style={"textAlign": "center", "marginTop": "10px"}),
                dcc.Interval(id="interval", interval=3000, n_intervals=0),  # auto-play
            ],
        ),

        # RIGHT COLUMN - Visitor Counts
        html.Div(
            style={"flex": "1.5", "padding": "20px", "background": "#f9fafc"},
            children=[
                html.H2("Visitor Counts 📈", style={"textAlign": "center"}),
                dcc.Graph(id="visitor-plot"),
                dcc.Dropdown(
                    id="sensor-selector",
                    options=[{"label": s, "value": s} for s in sensor_cols],
                    value=sensor_cols[0],
                    multi=False,
                    placeholder="Select a sensor ID",
                ),
            ],
        ),
    ],
)

# ------------------------------------------------------------
# 5. Callbacks
# ------------------------------------------------------------

# Sync slider with timestamp display
@app.callback(
    Output("time-display", "children"),
    Input("time-slider", "value"),
)
def update_time_display(idx):
    timestamp = vis_df.index[idx]
    return f"Current Time: {timestamp}"

# Auto-advance the slider
@app.callback(
    Output("time-slider", "value"),
    Input("interval", "n_intervals"),
)
def auto_advance(n):
    return (n) % len(vis_df)

# Update heatmap + line plot
@app.callback(
    [Output("heatmap", "figure"), Output("visitor-plot", "figure")],
    [Input("time-slider", "value"), Input("sensor-selector", "value")],
)
def update_dashboard(idx, selected_sensor):
    timestamp = vis_df.index[idx]
    row = vis_df.iloc[idx]

    # --- HEATMAP ---
    loc_df_copy = loc_df.copy()
    # Match sensor id prefix before "_"
    loc_df_copy["value"] = loc_df_copy["sensor_id"].map(
        lambda s: row[[c for c in row.index if c.startswith(s)]].mean()
        if any(c.startswith(s) for c in row.index)
        else 0
    )

    heatmap = px.scatter_mapbox(
        loc_df_copy,
        lat="latitude",
        lon="longitude",
        color="value",
        size="value",
        hover_name="location_name",
        color_continuous_scale="OrRd",
        zoom=13,
        center={"lat": 52.3728, "lon": 4.8936},
        mapbox_style="carto-positron",
    )
    heatmap.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # --- LINE PLOT ---
    df_sensor = vis_df[[selected_sensor]].copy()
    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=df_sensor.index,
            y=df_sensor[selected_sensor],
            mode="lines",
            name=selected_sensor,
        )
    )
    fig_line.add_vline(
        x=timestamp,
        line_width=2,
        line_dash="dash",
        line_color="orange",
        annotation_text="Now",
    )
    fig_line.update_layout(
        xaxis_title="Time",
        yaxis_title="Visitor Flow Count",
        margin={"l": 40, "r": 20, "t": 30, "b": 30},
    )

    return heatmap, fig_line

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
