# ---- Import required libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Load CSV files ----
file_june = "DIC_d13C_June2025.csv"
file_aug  = "DIC_d13C_Aug2025.csv"

df_june = pd.read_csv(file_june)
df_aug  = pd.read_csv(file_aug)

# ---- Ensure numeric and sort ----
for df in [df_june, df_aug]:
    for col in ["Depth (m)", "C (mg)", "d13C"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.sort_values("Depth (m)", inplace=True)
    df.reset_index(drop=True, inplace=True)

# ---- Extract arrays ----
z_june    = df_june["Depth (m)"].values
C_june    = df_june["C (mg)"].values
d13C_june = df_june["d13C"].values

z_aug     = df_aug["Depth (m)"].values
C_aug     = df_aug["C (mg)"].values
d13C_aug  = df_aug["d13C"].values

# ---- Plotting ----
fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)

# Depth vs C
ax[0].plot(C_june, z_june, marker='o', linestyle='-', color='purple', label="June")
ax[0].plot(C_aug, z_aug,   marker='s', linestyle='-', color='teal',   label="August")
ax[0].invert_yaxis()
ax[0].set_xlabel("C (mg)", fontsize=14)
ax[0].set_ylabel("Depth (m)", fontsize=14)

# Depth vs d13C
ax[1].plot(d13C_june, z_june, marker='o', linestyle='-', color='purple', label="June")
ax[1].plot(d13C_aug, z_aug,   marker='s', linestyle='-', color='teal',   label="August")
ax[1].set_xlabel("δ13C (‰)", fontsize=14)

# X-axis labels on top
for a in ax:
    a.xaxis.set_label_position('top')
    a.xaxis.tick_top()
    a.minorticks_on()
    a.yaxis.minorticks_on()
# ---- Shared legend at bottom center ----
fig.legend(['June', 'August'], loc='lower right', ncol =2, fontsize=14)
plt.tight_layout(rect=[0, 0.08, 1, 1])
# plt.suptitle("Depth Profiles: C and δ13C (June vs August 2025)", fontsize=14, weight='bold')
plt.show()
