import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator  # Correct import for minor ticks

# ---- File names ----
file1 = "Anion_June_2025.csv"    # CSV file for June anion data
file2 = "Anion_August_2025.csv"  # CSV file for August anion data
depth_column = "Depth"           # Column name for depth (change if different)

# ---- Map element column name to chemical formula ----
formulas = {
    "Chloride": "Cl⁻",
    "Nitrate": "NO₃⁻",
    "Sulfate": "SO₄²⁻",
    "Phosphate": "PO₄³⁻",
    "Fluoride": "F⁻",
    "Bromide": "Br⁻",
    "Thiosulfate": "S₂O₃²⁻",
    "Nitrite": "NO₂⁻",
    # add more if needed
}

# ---- Load CSV files into pandas DataFrames ----
df1 = pd.read_csv(file1)  # Load June data
df2 = pd.read_csv(file2)  # Load August data

# ---- Ensure Depth column is numeric ----
df1[depth_column] = pd.to_numeric(df1[depth_column], errors='coerce')
df2[depth_column] = pd.to_numeric(df2[depth_column], errors='coerce')
df1.dropna(subset=[depth_column], inplace=True)
df2.dropna(subset=[depth_column], inplace=True)

# ---- Identify element columns common to both CSVs ----
elements = [col for col in df1.columns if col != depth_column and col in df2.columns]

# ---- Create subplots ----
n = len(elements)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=True)
if n == 1:
    axes = [axes]  # Ensure axes is a list even if only one element

# Initialize variables to store handles and labels for a combined legend
handles_all, labels_all = None, None

# ---- Loop over each element to plot depth profiles ----
for i, elem in enumerate(elements):
    ax = axes[i]

    # Plot June data (solid line with circle markers)
    h1, = ax.plot(df1[elem], df1[depth_column], '-o',
                  color='purple', markeredgecolor='purple', label="June")

    # Plot August data (dashed line with square markers)
    h2, = ax.plot(df2[elem], df2[depth_column], '--s',
                  color='teal', markeredgecolor='teal', label="August")

    # Store handles and labels once for a single combined legend
    if handles_all is None:
        handles_all = [h1, h2]
        labels_all = ["June", "August"]

    # ---- Customize axes ----
    ax.invert_yaxis()                     # Depth increases downward
    ax.xaxis.set_label_position('top')    # Move X-axis label to top
    ax.xaxis.tick_top()                   # Move X-axis ticks to top
    ax.set_xlabel("ppm (mg/L)")           # X-axis label
    if i == 0:
        ax.set_ylabel("Depth (m)")        # Y-axis label on first subplot only

    # ---- Add minor Y-axis ticks outside ----
    ax.yaxis.set_minor_locator(AutoMinorLocator())  # Auto-generate minor ticks
    ax.tick_params(axis='y', which='minor', direction='out', length=4)

    # ---- Add element name and chemical formula to title ----
    formula = formulas.get(elem, "")                  # Lookup formula
    title = elem if formula == "" else f"{elem} ({formula})"
    ax.set_title(title, pad=30)                       # Add title with padding

    # ---- Grid lines ----
    ax.grid(True, which='major', color='gray', alpha=0.15)  # Faded major grid lines
    ax.grid(False, which='minor')                           # Minor grid lines off

# ---- Add a single combined legend above all subplots ----
fig.legend(handles_all, labels_all, ncol=2)  # Place above plots

# ---- Layout adjustments ----
plt.tight_layout()  # Adjust subplot spacing automatically
plt.suptitle("Depth vs. Anion Concentrations – June vs August 2025", fontsize=14, y=1.05)

# ---- Display the figure ----
plt.show()
