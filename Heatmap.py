GEOJSON_URL = "https://maps.amsterdam.nl/open_geodata/geojson_lnglat.php?KAARTLAAG=INDELING_STADSDEEL&THEMA=gebiedsindeling"

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd


df = pd.read_csv("sensor-location.xlsx - Sheet1.csv")
print(df.head())

#Split the coordinates
df[["Lat", "Long"]] = df["Lat/Long"].str.split(",", expand=True)

#Convert to numeric
df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
df["Long"] = pd.to_numeric(df["Long"], errors="coerce")

# # Plot
# plt.figure(figsize=(6,6))
# plt.scatter(df["Long"], df["Lat"], c="red", marker="o")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Sensor locaties")
# plt.grid(True)
# plt.show()

stadsdelen = gpd.read_file(GEOJSON_URL)

# Plot
fig, ax = plt.subplots(figsize=(8,8))
stadsdelen.boundary.plot(ax=ax, color="black", linewidth=0.7)   # de kaart
ax.scatter(df["Long"], df["Lat"], c="red", marker="o", label="Sensoren")  # sensoren
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Sensorlocaties op Amsterdam-kaart")
ax.legend()
plt.show()

