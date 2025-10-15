import pandas as pd
import json
import re
import binascii
import io
import fastparquet

tomtom = pd.read_parquet('Data-share/Data-share/20250820163000_stream.tomtom.analyze-sail.parquet', engine='fastparquet')
vessels = pd.read_parquet('Data-share\Data-share/20250820163000_stream.vessel-positions-anonymized-processed.analyze-sail.parquet', engine='fastparquet')
sensors = pd.read_csv('TIL6022-group-project/sensordata_SAIL2025.csv')


def decode_tomtom(v):
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        cleaned = re.sub(r"[^\x20-\x7E]+", "", v)
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        return json.loads(cleaned[start:end])
    except Exception:
        return None
    
import re, json, binascii

def decode_vessels(v):
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        # keep only hex chars
        hex_str = re.sub(r'[^0-9A-Fa-f]', '', v)
        if not hex_str:
            return None

        # decode from hex → utf-8
        decoded = binascii.unhexlify(hex_str).decode('utf-8', errors='ignore')

        # trim any stray control chars
        decoded = re.sub(r'[^\x20-\x7E]+', '', decoded)

        # locate JSON portion
        start, end = decoded.find('{'), decoded.rfind('}') + 1
        if start == -1 or end <= start:
            return None

        return json.loads(decoded[start:end])
    except Exception:
        return None


# Decode TomTom messages
tomtom["decoded"] = tomtom["_value"].apply(decode_tomtom)
decoded_tomtom = pd.json_normalize(tomtom["decoded"].dropna())
expanded_tomtom = pd.concat([tomtom.drop(columns=["decoded"]), decoded_tomtom], axis=1)

# Decode vessel messages
vessels["decoded"] = vessels["_value"].apply(decode_vessels)
decoded_vessels = pd.json_normalize(vessels["decoded"].dropna())
expanded_vessels = pd.concat([vessels.drop(columns=["decoded"]), decoded_vessels], axis=1)

# # # Preview results
# print(expanded_tomtom.columns, expanded_tomtom.head())
# print(expanded_vessels.columns, expanded_vessels.head())

# --- VESSELS ---
# Keep only current position, timestamp, vessel size, and coordinates
vessels_data = expanded_vessels[[
    "upload-timestamp",
    "imo-number",   # when the record was uploaded
    "length",             # vessel length
    "lat",                # latitude
    "lon"                 # longitude
]].copy()

# Rename for clarity
vessels_data = vessels_data.rename(columns={
    "upload-timestamp": "timestamp",
})

# Drop rows without coordinates 
vessels_data = vessels_data.dropna(subset=["lat", "lon"])

print("vessels_data created:", vessels_data.shape)
print(vessels_data.head())


# --- TOMTOM ---
# Keep only the traffic info you care about
tomtom_data = expanded_tomtom[[
    "time",               # timestamp of traffic snapshot
    "data"                # contains id + traffic_level pairs
]].copy()

# Parse the inner mini-CSV from each TomTom row
import io

def parse_tomtom_data(row):
    try:
        df = pd.read_csv(io.StringIO(row["data"]))
        df["snapshot_time"] = pd.to_datetime(row["time"])
        return df
    except Exception:
        return pd.DataFrame()

# Expand all TomTom mini-tables into one combined DataFrame
traffic_parts = [parse_tomtom_data(row) for _, row in tomtom_data.iterrows()]
tomtom_data = pd.concat(traffic_parts, ignore_index=True)

print("tomtom_data created:", tomtom_data.shape)
print(tomtom_data.head())

vessels_data.to_parquet("vessels_data.parquet", index=False)
tomtom_data.to_parquet("tomtom_data.parquet", index=False)






























# # ------------------------------------------------------------
# # 1. Load and clean sensor location data
# # ------------------------------------------------------------
# def load_sensor_locations(path="sensor-location.xlsx - Sheet1.csv"):
#     df = pd.read_csv(path)
#     df = df.rename(
#         columns={
#             "Objectummer": "sensor_id",
#             "Locatienaam": "location_name",
#             "Lat/Long": "lat_long",
#             "Breedte": "width_m",
#             "Effectieve  breedte": "effective_width_m",
#         }
#     )
#     lat_lon_split = df["lat_long"].str.split(",", expand=True)
#     df["latitude"] = lat_lon_split[0].astype(float)
#     df["longitude"] = lat_lon_split[1].astype(float)

#     # normalize commas in width columns
#     for col in ["width_m", "effective_width_m"]:
#         df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)

#     df = df.drop(columns=["lat_long"])
#     return df

# # ------------------------------------------------------------
# # 2. Load visitor data (confidential dataset)
# # ------------------------------------------------------------
# def load_visitor_data(path="sensordata_SAIL2025.csv"):
#     df = pd.read_csv(path, parse_dates=["timestamp"])
#     df = df.set_index("timestamp").sort_index()
#     return df

# # ------------------------------------------------------------
# # 3. Prepare data
# # ------------------------------------------------------------
# loc_df = load_sensor_locations()
# vis_df = load_visitor_data()

# # Extract sensor columns (ignore time features)
# sensor_cols = [c for c in vis_df.columns if "_" in c]

# start_time = pd.Timestamp("2025-08-20 00:00:00+02:00")
# end_time = vis_df.index.max()

# # ------------------------------------------------------------
# # 4. Build Dash app
# # ------------------------------------------------------------
# app = Dash(__name__)

# app.layout = html.Div(
#     style={"display": "flex", "flexDirection": "row", "height": "100vh"},
#     children=[
#         # LEFT COLUMN - Weather Placeholder
#         html.Div(
#             style={"flex": "1", "padding": "20px", "background": "#f0f4f7"},
#             children=[
#                 html.H2("Weather Forecast ☁️", style={"textAlign": "center"}),
#                 html.Div("Coming soon...", style={"textAlign": "center", "fontSize": "18px"}),
#             ],
#         ),

#         # MIDDLE COLUMN - Map
#         html.Div(
#             style={"flex": "2", "padding": "10px"},
#             children=[
#                 html.H2("Sensor Locations - Heatmap 🌍", style={"textAlign": "center"}),
#                 dcc.Graph(id="heatmap"),
#                 dcc.Slider(
#                     min=0,
#                     max=len(vis_df) - 1,
#                     step=1,
#                     value=0,
#                     id="time-slider",
#                     marks=None,
#                     tooltip={"always_visible": True},
#                     updatemode="drag",
#                 ),
#                 html.Div(id="time-display", style={"textAlign": "center", "marginTop": "10px"}),
#                 dcc.Interval(id="interval", interval=3000, n_intervals=0),  # auto-play
#             ],
#         ),

#         # RIGHT COLUMN - Visitor Counts
#         html.Div(
#             style={"flex": "1.5", "padding": "20px", "background": "#f9fafc"},
#             children=[
#                 html.H2("Visitor Counts 📈", style={"textAlign": "center"}),
#                 dcc.Graph(id="visitor-plot"),
#                 dcc.Dropdown(
#                     id="sensor-selector",
#                     options=[{"label": s, "value": s} for s in sensor_cols],
#                     value=sensor_cols[0],
#                     multi=False,
#                     placeholder="Select a sensor ID",
#                 ),
#             ],
#         ),
#     ],
# )

# # ------------------------------------------------------------
# # 5. Callbacks
# # ------------------------------------------------------------

# # Sync slider with timestamp display
# @app.callback(
#     Output("time-display", "children"),
#     Input("time-slider", "value"),
# )
# def update_time_display(idx):
#     timestamp = vis_df.index[idx]
#     return f"Current Time: {timestamp}"

# # Auto-advance the slider
# @app.callback(
#     Output("time-slider", "value"),
#     Input("interval", "n_intervals"),
# )
# def auto_advance(n):
#     return (n) % len(vis_df)

# # Update heatmap + line plot
# @app.callback(
#     [Output("heatmap", "figure"), Output("visitor-plot", "figure")],
#     [Input("time-slider", "value"), Input("sensor-selector", "value")],
# )
# def update_dashboard(idx, selected_sensor):
#     timestamp = vis_df.index[idx]
#     row = vis_df.iloc[idx]

#     # --- HEATMAP ---
#     loc_df_copy = loc_df.copy()
#     # Match sensor id prefix before "_"
#     loc_df_copy["value"] = loc_df_copy["sensor_id"].map(
#         lambda s: row[[c for c in row.index if c.startswith(s)]].mean()
#         if any(c.startswith(s) for c in row.index)
#         else 0
#     )

#     heatmap = px.scatter_mapbox(
#         loc_df_copy,
#         lat="latitude",
#         lon="longitude",
#         color="value",
#         size="value",
#         hover_name="location_name",
#         color_continuous_scale="OrRd",
#         zoom=13,
#         center={"lat": 52.3728, "lon": 4.8936},
#         mapbox_style="carto-positron",
#     )
#     heatmap.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

#     # --- LINE PLOT ---
#     df_sensor = vis_df[[selected_sensor]].copy()
#     fig_line = go.Figure()
#     fig_line.add_trace(
#         go.Scatter(
#             x=df_sensor.index,
#             y=df_sensor[selected_sensor],
#             mode="lines",
#             name=selected_sensor,
#         )
#     )
#     fig_line.add_vline(
#         x=timestamp,
#         line_width=2,
#         line_dash="dash",
#         line_color="orange",
#         annotation_text="Now",
#     )
#     fig_line.update_layout(
#         xaxis_title="Time",
#         yaxis_title="Visitor Flow Count",
#         margin={"l": 40, "r": 20, "t": 30, "b": 30},
#     )

#     return heatmap, fig_line

# # ------------------------------------------------------------
# # Run
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     app.run_server(debug=True)
