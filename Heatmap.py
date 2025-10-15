import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("sensor-location.xlsx - Sheet1.csv")
print(df.head())

# Splits de Lat/Long kolom
df[["Lat", "Long"]] = df["Lat/Long"].str.split(",", expand=True)

# Zet om naar numeriek
df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
df["Long"] = pd.to_numeric(df["Long"], errors="coerce")

# Plot
plt.figure(figsize=(6,6))
plt.scatter(df["Long"], df["Lat"], c="red", marker="o")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Sensor locaties")
plt.grid(True)
plt.show()
