#!/usr/bin/env python3
"""
Enhanced Lake Stratification Analysis with Original Graph Styling
Calculates thermal stratification metrics and lake layer classification
This script analyzes lake profile data to determine thermal stratification,
identify lake layers, and calculate key limnological parameters.
"""

# ---- Import required libraries ----
import pandas as pd                     # For reading CSV files and manipulating tabular data structures
import numpy as np                      # For numerical operations, mathematical functions, and array handling
import matplotlib.pyplot as plt         # For creating plots, graphs, and data visualizations
from scipy.signal import savgol_filter  # For smoothing noisy data using Savitzky-Golay filter algorithm
from scipy.interpolate import interp1d  # For interpolating between data points (e.g., finding euphotic depth)
import matplotlib.patches as mpatches   # For creating legend patches and colored areas in plots

# Physical constants for stratification calculations
g = 9.81  # Gravitational acceleration in m/s² - used in buoyancy frequency calculations
rho0 = 1000.0  # Reference density for freshwater in kg/m³ - standard at 4°C

def freshwater_density(T):
    """
    Calculate freshwater density (kg/m³) as a function of temperature (°C)
    Uses UNESCO equation of state approximation for freshwater
    Temperature affects water density - colder water is denser than warmer water
    This relationship is fundamental to thermal stratification in lakes
    
    Parameters:
    T: Temperature in degrees Celsius (can be array or single value)
    
    Returns:
    Density in kg/m³
    """
    T = np.asarray(T)                   # Convert input to numpy array for vectorized operations
    # UNESCO equation: density decreases with increasing temperature in a nonlinear fashion
    return 1000 * (1 - ((T + 288.9414) / 508929.2) * (T - 3.9863) ** 2)

def compute_buoyancy_frequency(depth, density):
    """
    Compute Brunt–Väisälä frequency profile N(z) [s⁻¹]
    This quantifies the strength of stratification at each depth
    Higher values indicate stronger density gradients and more stable stratification
    Used to identify thermoclines and assess mixing potential
    
    Parameters:
    depth: Array of depth measurements in meters
    density: Array of water density values in kg/m³
    
    Returns:
    z_mid: Mid-point depths for gradient calculations
    N: Buoyancy frequency values in s⁻¹
    """
    if len(depth) < 2:                  # Need at least 2 points to calculate gradients
        return np.array([]), np.array([])  # Return empty arrays if insufficient data
    
    dz = np.diff(depth)                 # Calculate depth differences between consecutive measurements
    drho = np.diff(density)             # Calculate density differences between consecutive measurements
    
    # Avoid division by zero errors
    dz[dz == 0] = 1e-6                  # Replace zero depth differences with very small number
    
    N2 = -(g / rho0) * (drho / dz)      # Calculate N² using Brunt-Väisälä frequency formula
    N2[N2 < 0] = 0                      # Set negative values to zero (unstable stratification)
    N = np.sqrt(N2)                     # Take square root to get buoyancy frequency N
    
    # Assign frequency values to mid-point depths between measurements
    z_mid = depth[:-1] + dz / 2.0       # Calculate mid-point depths for gradient representation
    return z_mid, N                     # Return depth and frequency arrays

def compute_schmidt_stability(depth, density):
    """
    Calculate Schmidt stability S [J/m²]
    Represents the mechanical energy needed to completely mix the water column
    Higher values indicate stronger stratification and greater resistance to mixing
    Important for understanding lake mixing dynamics and seasonal turnover
    
    Parameters:
    depth: Array of depth measurements in meters
    density: Array of water density values in kg/m³
    
    Returns:
    Schmidt stability in J/m²
    """
    if len(depth) < 2:                  # Need multiple points for integration
        return 0                        # Return zero if insufficient data
    
    h = depth.max() - depth.min()       # Calculate total depth range of measurements
    if h <= 0:                          # Check for valid depth range
        return 0                        # Return zero if no depth variation
    
    # Calculate average density over the entire water column using trapezoidal integration
    rho_bar = np.trapz(density, depth) / h  # Volume-weighted average density
    # Calculate center of volume (depth-weighted average position)
    z_bar = np.trapz(density * depth, depth) / np.trapz(density, depth)
    
    # Schmidt stability integrand: energy required to move water masses to uniform distribution
    integrand = (z_bar - depth) * (density - rho_bar) / rho_bar
    S = g * np.trapz(integrand, depth)  # Integrate to get total energy required for complete mixing
    return max(0, S)                    # Ensure non-negative result (physics constraint)

# ---- Load CSV data file ----
df = pd.read_csv("Steel_Lake_Profile_June_Data.csv")  # Read lake profile data from CSV file into pandas DataFrame

# ---- Ensure numeric data types & handle missing values ----
# Define columns that should contain numeric data
numeric_cols = ["Depth", "°C", "Light", "DO-mg/L", "SPC-uS/cm", "Secchi"]
for col in numeric_cols:                # Loop through each column that should be numeric
    # Convert to numeric, replacing any non-numeric values with NaN (Not a Number)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values using forward and backward filling methods
df.ffill(inplace=True)                  # Forward-fill: replace NaN with previous valid value
df.bfill(inplace=True)                  # Backward-fill: replace remaining NaNs with next valid value

# Sort data by depth in ascending order (surface to bottom) and reset row indices
df = df.sort_values("Depth").reset_index(drop=True)

# ---- Extract measurement arrays from DataFrame ----
z = df["Depth"].values                  # Depth measurements in meters (independent variable)
T = df["°C"].values                     # Temperature measurements in degrees Celsius
DO = df["DO-mg/L"].values               # Dissolved oxygen concentration in mg/L
PAR = df["Light"].values                # Photosynthetically Active Radiation (light intensity)
COND = df["SPC-uS/cm"].values           # Specific conductivity in microsiemens per centimeter
secchi = df["Secchi"].iloc[0]           # Secchi depth - single value representing water clarity

# ---- Safe data smoothing function ----
def smooth_data(arr):
    """
    Smooth noisy data using Savitzky-Golay filter with comprehensive error handling
    This removes measurement noise while preserving important features like thermoclines
    
    Parameters:
    arr: Array of measurements to be smoothed
    
    Returns:
    Smoothed array of same length as input
    """
    arr = np.array(arr, dtype=float)               # Convert input to numpy float array
    arr = np.nan_to_num(arr, nan=np.nanmean(arr)) # Replace any NaN values with array mean
    n = len(arr)                                   # Get number of data points
    if n < 3:                                      # Need minimum 3 points for smoothing
        return arr                                 # Return original if too few points
    
    # Determine appropriate window length (must be odd number and <= total points)
    window_length = min(5, n if n % 2 == 1 else n-1)  # Use 5 points or adjust to odd number
    polyorder = min(2, window_length - 1)         # Polynomial order must be less than window length
    
    try:
        # Apply Savitzky-Golay filter: fits local polynomials to smooth data
        return savgol_filter(arr, window_length, polyorder)
    except:
        return arr                                 # Return original data if smoothing fails

# Apply smoothing to temperature and dissolved oxygen profiles
T_s = smooth_data(T)                    # Smoothed temperature profile
DO_s = smooth_data(DO)                  # Smoothed dissolved oxygen profile

# Calculate water density from temperature using physical relationship
density = freshwater_density(T_s)      # Density array calculated from smoothed temperature

# ---- Enhanced Thermocline Detection and Metalimnion Boundaries ----
# Calculate temperature gradient with respect to depth
dTdz = np.gradient(T_s, z)              # Temperature change per unit depth (°C/m)
# Find depth where temperature gradient is steepest (absolute maximum)
thermo_depth = z[np.argmax(np.abs(dTdz))]  # Thermocline depth in meters
max_gradient = np.max(np.abs(dTdz))     # Maximum temperature gradient magnitude

# Define metalimnion (transition layer) based on significant temperature gradients
threshold = 0.2 * max_gradient          # Set threshold at 20% of maximum gradient
meta_mask = np.abs(dTdz) >= threshold   # Boolean array: True where gradient exceeds threshold

# Determine metalimnion boundaries based on gradient analysis
if np.any(meta_mask):                   # If significant gradients are found
    meta_top = z[meta_mask].min()       # Shallowest depth with significant gradient
    meta_bottom = z[meta_mask].max()    # Deepest depth with significant gradient
else:                                   # If no strong gradients detected
    meta_top = thermo_depth             # Use single thermocline depth for both boundaries
    meta_bottom = thermo_depth
    print("Warning: no strong thermocline detected. Assigning single thermocline depth.")

# ---- Calculate Enhanced Stratification Metrics ----
# Compute buoyancy frequency profile for quantitative stratification assessment
z_mid, N = compute_buoyancy_frequency(z, density)  # Mid-depths and buoyancy frequencies

# Calculate Schmidt stability - energy required for complete mixing
schmidt_stability = compute_schmidt_stability(z, density)  # Stability in J/m²

# Calculate surface-to-bottom temperature difference
temp_difference = abs(T_s[0] - T_s[-1])  # Absolute temperature difference in °C

# Classify stratification strength based on temperature difference and gradient
if temp_difference > 5.0 and max_gradient > 1.0:      # Strong thermal stratification
    stratification_strength = "Strong"
elif temp_difference > 2.0 and max_gradient > 0.5:    # Moderate thermal stratification
    stratification_strength = "Moderate"
elif temp_difference > 1.0:                           # Weak but detectable stratification
    stratification_strength = "Weak"
else:                                                  # Well-mixed conditions
    stratification_strength = "Well-mixed"

# ---- Analyze Oxygen Distribution and Identify Oxycline ----
# Calculate dissolved oxygen gradient with depth
dDOdz = np.gradient(DO_s, z)            # DO change per unit depth (mg/L/m)
# Find depth of steepest oxygen gradient (oxycline)
oxycline_depth = z[np.argmax(np.abs(dDOdz))]  # Oxycline depth in meters

# Identify depths with low oxygen conditions
hypoxic = z[DO < 2.0]                   # Depths where DO < 2 mg/L (hypoxic threshold)
anoxic = z[DO < 0.5]                    # Depths where DO < 0.5 mg/L (anoxic threshold)

# ---- Calculate Light Penetration and Euphotic Depth ----
if np.all(PAR > 0):                     # If all PAR measurements are positive (valid)
    surface_PAR = PAR[0]                # Surface light intensity (reference level)
    # Create interpolation function to find depth where light drops to 1% of surface
    f = interp1d(PAR[::-1], z[::-1], bounds_error=False, fill_value=np.nan)  # Reverse arrays for interpolation
    try:
        # Find depth where PAR equals 1% of surface value (euphotic zone boundary)
        z_eu = float(f(0.01 * surface_PAR))  # Euphotic depth in meters
    except:
        # Use empirical relationship if interpolation fails
        z_eu = 1.7 * secchi             # 1.7 times Secchi depth (standard approximation)
else:
    # Use Secchi depth relationship if PAR data unavailable
    z_eu = 1.7 * secchi                 # Fallback euphotic depth calculation

# ---- Identify Chemocline from Conductivity Profile ----
# Calculate conductivity gradient with depth
dCdz = np.gradient(COND, z)             # Conductivity change per unit depth (μS/cm/m)
# Find depth of maximum conductivity gradient (chemocline)
chem_depth = z[np.argmax(np.abs(dCdz))] # Chemocline depth in meters

# ---- Assign Lake Layer Classifications ----
layers = []                             # Initialize list to store layer names
for depth in z:                         # Loop through each depth measurement
    if depth < meta_top:                # Above metalimnion
        layers.append("epilimnion")     # Upper mixed layer (warm, oxygenated)
    elif meta_top <= depth <= meta_bottom:  # Within metalimnion boundaries
        layers.append("metalimnion")    # Transition layer (rapid temperature/density change)
    else:                               # Below metalimnion
        layers.append("hypolimnion")    # Deep layer (cold, potentially low oxygen)

# ---- Create Boolean Flags for Environmental Zones ----
# Euphotic zone flag: True for depths receiving sufficient light for photosynthesis
euphotic_flag = [depth <= z_eu for depth in z]
# Hypoxic zone flag: True for depths below hypoxia onset
hypoxic_flag = [depth >= hypoxic.min() if len(hypoxic) > 0 else False for depth in z]
# Anoxic zone flag: True for depths below anoxia onset
anoxic_flag = [depth >= anoxic.min() if len(anoxic) > 0 else False for depth in z]

# ---- Generate Comments for Key Transition Depths ----
comments = []                           # Initialize list for depth-specific comments
for depth, layer in zip(z, layers):    # Loop through depths and corresponding layers
    c = ""                              # Initialize empty comment string
    # Check if current depth is near thermocline (within 0.2 m tolerance)
    if np.isclose(depth, thermo_depth, atol=0.2):
        c += "Thermocline "             # Add thermocline annotation
    # Check if current depth is near oxycline (within 0.2 m tolerance)
    if np.isclose(depth, oxycline_depth, atol=0.2):
        c += "Oxycline "                # Add oxycline annotation
    # Check if current depth is near chemocline (within 0.2 m tolerance)
    if np.isclose(depth, chem_depth, atol=0.2):
        c += "Chemocline "              # Add chemocline annotation
    comments.append(c.strip())          # Remove trailing spaces and add to list

# ---- Add New Analysis Columns to DataFrame ----
df["layer"] = layers                    # Lake layer classification for each depth
df["euphotic"] = euphotic_flag          # Boolean: within euphotic zone
df["hypoxic"] = hypoxic_flag            # Boolean: within hypoxic zone
df["anoxic"] = anoxic_flag              # Boolean: within anoxic zone
df["comments"] = comments               # Text annotations for key depths
df["temperature_gradient"] = dTdz       # Temperature gradient (°C/m)
df["density"] = density                 # Calculated water density (kg/m³)
df["smoothed_temp"] = T_s               # Smoothed temperature profile
df["smoothed_DO"] = DO_s                # Smoothed dissolved oxygen profile

# ---- Print Comprehensive Analysis Summary ----
print("\n--- Enhanced Lake Layer Classification ---")
print(f"Thermocline depth: {thermo_depth:.2f} m")                    # Depth of maximum temperature gradient
print(f"Maximum temperature gradient: {max_gradient:.3f} °C/m")      # Steepest temperature change rate
print(f"Metalimnion: {meta_top:.2f} – {meta_bottom:.2f} m")          # Transition layer boundaries
print(f"Surface-bottom temperature difference: {temp_difference:.2f} °C")  # Total thermal stratification
print(f"Schmidt stability: {schmidt_stability:.2f} J/m²")            # Energy needed for complete mixing
print(f"Stratification strength: {stratification_strength}")         # Qualitative assessment
print(f"Oxycline depth: {oxycline_depth:.2f} m")                     # Depth of steepest oxygen gradient
if len(hypoxic) > 0:                    # If hypoxic conditions detected
    print(f"Hypoxic zone starts at {hypoxic.min():.2f} m")           # Shallowest hypoxic depth
if len(anoxic) > 0:                     # If anoxic conditions detected
    print(f"Anoxic zone starts at {anoxic.min():.2f} m")             # Shallowest anoxic depth
print(f"Euphotic depth: {z_eu:.2f} m")                               # Light penetration limit
print(f"Chemocline depth: {chem_depth:.2f} m")                       # Conductivity gradient maximum
if len(N) > 0:                          # If buoyancy frequency calculated successfully
    print(f"Maximum buoyancy frequency: {N.max():.4f} s⁻¹ at {z_mid[np.argmax(N)]:.2f} m")  # Strongest stratification point

# ---- Save Enhanced Analysis Results ----
df.to_csv("Lake_Layer_Output.csv", index=False)  # Save complete DataFrame with all new columns
print("\nOutput saved as Lake_Layer_Output.csv")  # Confirmation message

# ---- Create Multi-Panel Profile Visualization ----
# Create figure with 4 subplots sharing the same y-axis (depth)
fig, ax = plt.subplots(1, 4, figsize=(14,6), sharey=True)

# Panel 1: Temperature Profile
ax[0].plot(T, z, 'o-', label="Temperature")                          # Plot temperature vs depth with markers
ax[0].axhline(thermo_depth, color="r", linestyle="--", label="Thermocline")  # Mark thermocline depth
ax[0].invert_yaxis()                    # Invert y-axis so surface (0m) is at top
ax[0].set_xlabel("Temperature (°C)")    # X-axis label
ax[0].set_ylabel("Depth (m)")           # Y-axis label (only for leftmost panel)

# Panel 2: Dissolved Oxygen Profile
ax[1].plot(DO, z, 'o-', label="DO mg/L")                            # Plot DO vs depth
ax[1].axhline(oxycline_depth, color="m", linestyle="--", label="Oxycline")  # Mark oxycline depth
ax[1].set_xlabel("Dissolved Oxygen (mg/L)")  # X-axis label

# Panel 3: Light Penetration Profile
ax[2].plot(PAR, z, 'o-', label="Light (PAR)")                       # Plot light vs depth
ax[2].axhline(z_eu, color="g", linestyle="--", label="Euphotic depth")  # Mark euphotic depth
ax[2].set_xlabel("PAR")                 # X-axis label

# Panel 4: Conductivity Profile
ax[3].plot(COND, z, 'o-', label="SPC-uS/cm")                        # Plot conductivity vs depth
ax[3].axhline(chem_depth, color="purple", linestyle="--", label="Chemocline")  # Mark chemocline
ax[3].set_xlabel("Conductivity (µS/cm)")  # X-axis label

# ---- Format All Subplot Axes ----
for a in ax:                            # Loop through all subplot axes
    a.xaxis.set_label_position('top')   # Move x-axis labels to top of plot
    a.xaxis.tick_top()                  # Move x-axis tick marks to top
    a.minorticks_on()                   # Enable minor tick marks on x-axis
    a.yaxis.minorticks_on()             # Enable minor tick marks on y-axis

# ---- Add Layer Shading and Oxygen Zone Indicators ----
for a in ax:                            # Apply to all subplots
    xlims = a.get_xlim()                # Get current x-axis limits for shading
    
    # Epilimnion shading (surface water to top of metalimnion)
    a.fill_betweenx([z.min(), meta_top], xlims[0], xlims[1],
                    color='lightgreen', alpha=0.2)  # Light green background
    
    # Metalimnion shading (transition layer between epilimnion and hypolimnion)
    a.fill_betweenx([meta_top, meta_bottom], xlims[0], xlims[1],
                    color='lightblue', alpha=0.2)   # Light blue background
    
    # Hypolimnion shading (deep water below metalimnion)
    a.fill_betweenx([meta_bottom, z.max()], xlims[0], xlims[1],
                    color='lightgray', alpha=0.2)   # Light gray background
    
    # Add horizontal lines for oxygen depletion zones
    if len(hypoxic) > 0:                # If hypoxic conditions exist
        a.axhline(hypoxic.min(), color='orange', linestyle=':')  # Orange dotted line
    if len(anoxic) > 0:                 # If anoxic conditions exist
        a.axhline(anoxic.min(), color='black', linestyle=':')   # Black dotted line

# ---- Create Custom Legend Elements ----
# Create colored patches for layer legend entries
epilim_patch = mpatches.Patch(color='lightgreen', alpha=0.2, label='Epilimnion')
metalim_patch = mpatches.Patch(color='lightblue', alpha=0.2, label='Metalimnion')
hypolim_patch = mpatches.Patch(color='lightgray', alpha=0.2, label='Hypolimnion')

# Create line elements for oxygen zone legend entries
hypoxic_line = plt.Line2D([0],[0], color='orange', linestyle=':', label='Hypoxic start')
anoxic_line = plt.Line2D([0],[0], color='black', linestyle=':', label='Anoxic start')

# ---- Add Comprehensive Legends to Each Subplot ----
for a in ax:                            # Loop through all subplots
    handles, labels = a.get_legend_handles_labels()  # Get existing legend elements
    # Add layer shading patches to legend
    new_handles = handles + [epilim_patch, metalim_patch, hypolim_patch]
    new_labels = labels + ['Epilimnion', 'Metalimnion', 'Hypolimnion']
    
    # Add oxygen zone lines if they exist in the data
    if len(hypoxic) > 0:                # Add hypoxic line to legend if hypoxia detected
        new_handles.append(hypoxic_line)
        new_labels.append('Hypoxic start')
    if len(anoxic) > 0:                 # Add anoxic line to legend if anoxia detected
        new_handles.append(anoxic_line)
        new_labels.append('Anoxic start')
    
    # Create legend with all elements positioned automatically
    a.legend(new_handles, new_labels, loc='best')

# ---- Finalize and Display Plot ----
plt.tight_layout()                      # Automatically adjust subplot spacing to prevent overlap
plt.show()                              # Display the complete multi-panel figure
