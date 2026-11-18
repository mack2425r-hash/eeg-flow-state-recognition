import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

INPUT_FILE = r"D:\paper\rf_balanced_feature_importance.csv"

OUTPUT_FILE = r"D:\paper\domain_importance.csv"

FIGURE_FILE = r"D:\paper\domain_importance.png"


# ============================================================
# FEATURE GROUPS
# ============================================================

TIME_FEATURES = [
    "Mean",
    "Variance",
    "Std",
    "RMS",
    "Hjorth_Activity",
    "Hjorth_Mobility",
    "Hjorth_Complexity"
]

FREQUENCY_FEATURES = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma",
    "Engagement_Index"
]

TIME_FREQUENCY_FEATURES = [
    "STFT_Energy",
    "Wavelet_Energy"
]

NONLINEAR_FEATURES = [
    "Shannon_Entropy",
    "Fractal_Dimension",
    "Hurst_Exponent"
]


# ============================================================
# LOAD IMPORTANCE
# ============================================================

data = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# ASSIGN DOMAIN
# ============================================================

def assign_domain(feature):

    if feature in TIME_FEATURES:
        return "Time Domain"

    elif feature in FREQUENCY_FEATURES:
        return "Frequency Domain"

    elif feature in TIME_FREQUENCY_FEATURES:
        return "Time-Frequency"

    elif feature in NONLINEAR_FEATURES:
        return "Non-Linear"

    else:
        return "Unknown"


data["Domain"] = data["Feature"].apply(
    assign_domain
)


# ============================================================
# CALCULATE DOMAIN IMPORTANCE
# ============================================================

domain_results = (
    data
    .groupby("Domain")
    .agg(
        Total_Importance=(
            "Importance",
            "sum"
        ),

        Average_Importance=(
            "Importance",
            "mean"
        ),

        Number_of_Features=(
            "Feature",
            "count"
        )
    )
    .reset_index()
)


# ============================================================
# SORT
# ============================================================

domain_results = domain_results.sort_values(
    by="Total_Importance",
    ascending=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("RESULT 8 - FEATURE DOMAIN IMPORTANCE")
print("==============================================\n")

print(
    domain_results.to_string(
        index=False
    )
)


# ============================================================
# SAVE TABLE
# ============================================================

domain_results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PLOT
# ============================================================

plot_data = domain_results.sort_values(
    by="Total_Importance",
    ascending=True
)


plt.figure(
    figsize=(8, 5)
)


plt.barh(
    plot_data["Domain"],
    plot_data["Total_Importance"]
)


plt.xlabel(
    "Total Random Forest Feature Importance"
)

plt.ylabel(
    "Feature Domain"
)

plt.title(
    "Contribution of EEG Feature Domains"
)


plt.tight_layout()


plt.savefig(
    FIGURE_FILE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# FINAL
# ============================================================

print("\n==============================================")
print("RESULT 8 COMPLETED")
print("==============================================")

print(
    "\nTable:",
    OUTPUT_FILE
)

print(
    "Figure:",
    FIGURE_FILE
)