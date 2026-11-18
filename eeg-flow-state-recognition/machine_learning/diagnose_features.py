import pandas as pd
import numpy as np


# ============================================================
# FEATURE DIAGNOSTIC ANALYSIS
# ============================================================

DATA_FILE = r"D:\paper\advanced_training_data.csv"

OUTPUT_FILE = r"D:\paper\feature_class_statistics.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n==============================================")
print("      EEG FEATURE DIAGNOSTIC ANALYSIS")
print("==============================================")

data = pd.read_csv(DATA_FILE)

print("\nDataset Shape:", data.shape)


# ============================================================
# 2. CHECK LABEL DISTRIBUTION
# ============================================================

print("\n==============================================")
print("CLASS DISTRIBUTION")
print("==============================================")

class_counts = data["Label"].value_counts().sort_index()

print(
    "Non-Flow (0):",
    class_counts.get(0, 0)
)

print(
    "Flow (1):",
    class_counts.get(1, 0)
)


# ============================================================
# 3. SEPARATE CLASSES
# ============================================================

non_flow = data[
    data["Label"] == 0
]

flow = data[
    data["Label"] == 1
]


# ============================================================
# 4. FEATURE LIST
# ============================================================

features = [
    "Mean",
    "Variance",
    "Std",
    "RMS",
    "Hjorth_Activity",
    "Hjorth_Mobility",
    "Hjorth_Complexity",
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma",
    "Engagement_Index",
    "STFT_Energy",
    "Wavelet_Energy",
    "Shannon_Entropy",
    "Fractal_Dimension",
    "Hurst_Exponent"
]


# ============================================================
# 5. BASIC FEATURE STATISTICS
# ============================================================

results = []


for feature in features:

    non_flow_mean = non_flow[feature].mean()
    flow_mean = flow[feature].mean()

    non_flow_std = non_flow[feature].std()
    flow_std = flow[feature].std()

    non_flow_median = non_flow[feature].median()
    flow_median = flow[feature].median()

    results.append({

        "Feature": feature,

        "NonFlow_Mean":
            non_flow_mean,

        "Flow_Mean":
            flow_mean,

        "NonFlow_Std":
            non_flow_std,

        "Flow_Std":
            flow_std,

        "NonFlow_Median":
            non_flow_median,

        "Flow_Median":
            flow_median

    })


statistics = pd.DataFrame(
    results
)


# ============================================================
# 6. EFFECT SIZE
# ============================================================

print("\n==============================================")
print("CALCULATING FEATURE SEPARATION")
print("==============================================")


effect_sizes = []


for feature in features:

    x = non_flow[feature]
    y = flow[feature]

    mean_difference = (
        y.mean() - x.mean()
    )

    pooled_std = np.sqrt(
        (
            x.std() ** 2
            +
            y.std() ** 2
        ) / 2
    )

    if pooled_std == 0:

        effect_size = 0.0

    else:

        effect_size = (
            mean_difference
            /
            pooled_std
        )

    effect_sizes.append(
        effect_size
    )


statistics["Effect_Size"] = (
    effect_sizes
)


statistics["Absolute_Effect_Size"] = (
    statistics["Effect_Size"].abs()
)


# ============================================================
# 7. CORRELATION WITH LABEL
# ============================================================

print("\n==============================================")
print("CALCULATING LABEL CORRELATION")
print("==============================================")


correlations = []


for feature in features:

    correlation = data[
        [feature, "Label"]
    ].corr().iloc[0, 1]

    correlations.append(
        correlation
    )


statistics["Label_Correlation"] = (
    correlations
)


statistics["Absolute_Correlation"] = (
    statistics["Label_Correlation"].abs()
)


# ============================================================
# 8. SORT BY EFFECT SIZE
# ============================================================

statistics = statistics.sort_values(
    by="Absolute_Effect_Size",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("FEATURE STATISTICS")
print("==============================================\n")


display_columns = [

    "Feature",

    "NonFlow_Mean",

    "Flow_Mean",

    "Effect_Size",

    "Label_Correlation"

]


print(
    statistics[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

statistics.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==============================================")
print("TOP FEATURES BY EFFECT SIZE")
print("==============================================\n")


for i, row in statistics.head(10).iterrows():

    print(
        f"{i + 1:2d}. "
        f"{row['Feature']:25s} "
        f"Effect Size = "
        f"{row['Effect_Size']:.4f}"
    )


# ============================================================
# 11. TOP FEATURES BY CORRELATION
# ============================================================

correlation_sorted = statistics.sort_values(
    by="Absolute_Correlation",
    ascending=False
)


print("\n==============================================")
print("TOP FEATURES BY LABEL CORRELATION")
print("==============================================\n")


for i, row in correlation_sorted.head(10).iterrows():

    print(
        f"{i + 1:2d}. "
        f"{row['Feature']:25s} "
        f"Correlation = "
        f"{row['Label_Correlation']:.4f}"
    )


# ============================================================
# 12. CHECK FOR CONSTANT FEATURES
# ============================================================

print("\n==============================================")
print("CONSTANT / NEAR-CONSTANT FEATURES")
print("==============================================\n")


for feature in features:

    unique_values = data[
        feature
    ].nunique()

    if unique_values <= 2:

        print(
            feature,
            "→",
            unique_values,
            "unique values"
        )


# ============================================================
# 13. CHECK FOR NaN / INFINITY
# ============================================================

print("\n==============================================")
print("DATA QUALITY CHECK")
print("==============================================")

print(
    "\nNaN values:"
)

print(
    data[features].isna().sum()
)


print(
    "\nInfinite values:"
)

print(
    np.isinf(
        data[features]
    ).sum()
)


# ============================================================
# 14. FINAL
# ============================================================

print("\n==============================================")
print("DIAGNOSTIC COMPLETED")
print("==============================================")

print(
    "\nSaved:",
    OUTPUT_FILE
)