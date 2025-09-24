#!/usr/bin/env python3
"""
Lake Profile Analysis & Plotting
- Computes thermal layers, oxycline, hypoxia/anoxia, euphotic depth, and chemocline.
- Plots 4-panel profile: Temperature, Dissolved Oxygen, PAR, Specific Conductivity.
- Uses teal lines with square markers for all profiles.
- Single shared legend below all subplots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches

# ---- Load CSV ----
df = pd.read_csv("Steel_Lake_Profile_June_Data.csv")

# ---- Ensure numeric & handle missing values ----
numeric_cols = ["Depth", "°C", "Light", "DO-mg/L", "SPC-uS/cm", "Secchi"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.ffill(inplace=True)
df.bfill(inplace=True)
df = df.sort_values("Depth").reset_index(drop=True)

# ---- Extract arrays ----
z = df["Depth"].values
T = df["°C"].values
DO = df["DO-mg/L"].values
PAR = df["Light"].values
COND = df["SPC-uS/cm"].values
secchi = df["Secchi"].iloc[0]

# ---- Smoothing function ----
def smooth_data(arr):
    arr = np.array(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=np.nanmean(arr))
    n = len(arr)
    if n < 3:
        return arr
    window_length = min(5, n if n % 2 == 1 else n-1)
    polyorder = min(2, window_length - 1)
    return savgol_filter(arr, window_length, polyorder)

T_s = smooth_data(T)
DO_s = smooth_data(DO)

# ---- Thermocline / metalimnion ----
dTdz = np.gradient(T_s, z)
thermo_depth = z[np.argmax(np.abs(dTdz))]

threshold = 0.2 * np.max(np.abs(dTdz))
meta_mask = np.abs(dTdz) >= threshold

meta_top = z[meta_mask].min() if np.any(meta_mask) else thermo_depth
meta_bottom = z[meta_mask].max() if np.any(meta_mask) else thermo_depth
if not np.any(meta_mask):
    print("Warning: no strong thermocline detected. Assigning single thermocline depth.")

# ---- Oxygen layers ----
dDOdz = np.gradient(DO_s, z)
oxycline_depth = z[np.argmax(np.abs(dDOdz))]
hypoxic = z[DO < 2.0]
anoxic = z[DO < 0.5]

# ---- Light / euphotic depth ----
if np.all(PAR > 0):
    surface_PAR = PAR[0]
    f = interp1d(PAR[::-1], z[::-1], bounds_error=False, fill_value=np.nan)
    try:
        z_eu = float(f(0.01 * surface_PAR))
    except:
        z_eu = 1.7 * secchi
else:
    z_eu = 1.7 * secchi

# ---- Conductivity / chemocline ----
dCdz = np.gradient(COND, z)
chem_depth = z[np.argmax(np.abs(dCdz))]

# ---- Layer assignment ----
layers = []
for depth in z:
    if depth < meta_top:
        layers.append("epilimnion")
    elif meta_top <= depth <= meta_bottom:
        layers.append("metalimnion")
    else:
        layers.append("hypolimnion")

# ---- Flags ----
euphotic_flag = [depth <= z_eu for depth in z]
hypoxic_flag = [depth >= hypoxic.min() if len(hypoxic) > 0 else False for depth in z]
anoxic_flag = [depth >= anoxic.min() if len(anoxic) > 0 else False for depth in z]

# ---- Comments for key depths ----
comments = []
for depth in z:
    c = ""
    if np.isclose(depth, thermo_depth, atol=0.2):
        c += "Thermocline "
    if np.isclose(depth, oxycline_depth, atol=0.2):
        c += "Oxycline "
    if np.isclose(depth, chem_depth, atol=0.2):
        c += "Chemocline "
    comments.append(c.strip())

# ---- Add columns ----
df["layer"] = layers
df["euphotic"] = euphotic_flag
df["hypoxic"] = hypoxic_flag
df["anoxic"] = anoxic_flag
df["comments"] = comments

# ---- Summary ----
print("\n--- Lake Layer Classification ---")
print(f"Thermocline depth: {thermo_depth:.2f} m")
print(f"Metalimnion: {meta_top:.2f} – {meta_bottom:.2f} m")
print(f"Oxycline depth: {oxycline_depth:.2f} m")
if len(hypoxic) > 0:
    print(f"Hypoxic zone starts at {hypoxic.min():.2f} m")
if len(anoxic) > 0:
    print(f"Anoxic zone starts at {anoxic.min():.2f} m")
print(f"Euphotic depth: {z_eu:.2f} m")
print(f"Chemocline depth: {chem_depth:.2f} m")

df.to_csv("Lake_Layer_Output.csv", index=False)
print("\nOutput saved as Lake_Layer_Output.csv")

# ---- Plotting ----
fig, ax = plt.subplots(1, 4, figsize=(14,6), sharey=True)
line_color = 'purple'
marker_style = 'o'

# Temperature
ax[0].plot(T, z, marker=marker_style, color=line_color)
ax[0].axhline(thermo_depth, color="r", linestyle="--")
ax[0].set_xlabel("Temperature (°C)")
ax[0].set_ylabel("Depth (m)")
ax[0].invert_yaxis()

# Dissolved Oxygen
ax[1].plot(DO, z, marker=marker_style, color=line_color)
ax[1].axhline(oxycline_depth, color="m", linestyle="--")
ax[1].set_xlabel("Dissolved Oxygen (mg/L)")

# PAR
ax[2].plot(PAR, z, marker=marker_style, color=line_color)
ax[2].axhline(z_eu, color="g", linestyle="--")
ax[2].set_xlabel("PAR")

# Specific Conductivity
ax[3].plot(COND, z, marker=marker_style, color=line_color)
ax[3].axhline(chem_depth, color="purple", linestyle="--")
ax[3].set_xlabel("Specific Conductivity (µS/cm)")

# X-axis labels on top, minor ticks
for a in ax:
    a.xaxis.set_label_position('top')
    a.xaxis.tick_top()
    a.minorticks_on()
    a.yaxis.minorticks_on()

# ---- Shading layers ----
for a in ax:
    xlims = a.get_xlim()
    a.fill_betweenx([z.min(), meta_top], xlims[0], xlims[1], color='lightgreen', alpha=0.2)
    a.fill_betweenx([meta_top, meta_bottom], xlims[0], xlims[1], color='lightblue', alpha=0.2)
    a.fill_betweenx([meta_bottom, z.max()], xlims[0], xlims[1], color='lightgray', alpha=0.2)
    if len(hypoxic) > 0:
        a.axhline(hypoxic.min(), color='orange', linestyle=':')
    if len(anoxic) > 0:
        a.axhline(anoxic.min(), color='black', linestyle=':')

# ---- Shared legend ----
epilim_patch = mpatches.Patch(color='lightgreen', alpha=0.2, label='Epilimnion')
metalim_patch = mpatches.Patch(color='lightblue', alpha=0.2, label='Metalimnion')
hypolim_patch = mpatches.Patch(color='lightgray', alpha=0.2, label='Hypolimnion')
thermo_line = plt.Line2D([0],[0], color='r', linestyle='--', label='Thermocline')
oxy_line = plt.Line2D([0],[0], color='m', linestyle='--', label='Oxycline')
euphotic_line = plt.Line2D([0],[0], color='g', linestyle='--', label='Euphotic depth')
chem_line = plt.Line2D([0],[0], color='purple', linestyle='--', label='Chemocline')
hypoxic_line = plt.Line2D([0],[0], color='orange', linestyle=':', label='Hypoxic start')
anoxic_line = plt.Line2D([0],[0], color='black', linestyle=':', label='Anoxic start')
teal_line = plt.Line2D([0],[0], color=line_color, marker=marker_style, label='June 2025')

all_handles = [teal_line, epilim_patch, metalim_patch, hypolim_patch,
               thermo_line, oxy_line, euphotic_line, chem_line]

if len(hypoxic) > 0:
    all_handles.append(hypoxic_line)
if len(anoxic) > 0:
    all_handles.append(anoxic_line)

fig.legend(handles=all_handles, loc='lower center', ncol=5, fontsize=12)
plt.tight_layout(rect=[0, 0.07, 1, 1])  # Leave space at bottom for legend
plt.show()
