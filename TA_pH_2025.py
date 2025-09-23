# ---- Import required libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Load CSV files ----
file_june = "TA_June2025.csv"
file_aug  = "TA_pH_Aug2025.csv"

df_june = pd.read_csv(file_june)
df_aug  = pd.read_csv(file_aug)

# ---- Ensure numeric and sort ----
# June: Depth(m), TA
# August: Depth(m), TA, pH
for df, cols in zip([df_june, df_aug], [["Depth(m)", "TA"], ["Depth(m)", "TA", "pH"]]):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.sort_values("Depth(m)", inplace=True)
    df.reset_index(drop=True, inplace=True)

# ---- Extract arrays ----
z_june = df_june["Depth(m)"].values
TA_june = df_june["TA"].values

z_aug = df_aug["Depth(m)"].values
TA_aug = df_aug["TA"].values
pH_aug = df_aug["pH"].values  # Only August has pH

# ---- Plotting ----
fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)

# Depth vs TA
ax[0].plot(TA_june, z_june, marker='o', linestyle='-', color='purple', label="June")
ax[0].plot(TA_aug, z_aug,   marker='s', linestyle='-', color='teal',   label="August")
ax[0].invert_yaxis()
ax[0].set_xlabel("Total Alkalinity (mg/L)", fontsize=14)
ax[0].set_ylabel("Depth (m)", fontsize=14)

# Depth vs pH (only August)
ax[1].plot(pH_aug, z_aug, marker='s', linestyle='-', color='teal', label="August")
ax[1].set_xlabel("pH", fontsize=14)

# X-axis labels on top
for a in ax:
    a.xaxis.set_label_position('top')
    a.xaxis.tick_top()
    a.minorticks_on()
    a.yaxis.minorticks_on()

# ---- Shared legend at bottom center ----
fig.legend(['June', 'August'], loc='lower right', ncol=2, fontsize=14)
plt.tight_layout(rect=[0, 0.08, 1, 1])
# plt.suptitle("Depth Profiles: TA and pH (June vs August 2025)", fontsize=14, weight='bold')
plt.show()

