import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = r"D:\paper\advanced_training_data.csv"

OUTPUT_FILE = r"D:\paper\statistical_feature_analysis.csv"


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [

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
# LOAD DATA
# ============================================================

print("\n")
print("=" * 70)
print("       RESULT 9 - STATISTICAL FEATURE ANALYSIS")
print("       FLOW vs NON-FLOW")
print("=" * 70)


data = pd.read_csv(
    DATA_FILE
)


print(
    "\nDataset Shape:",
    data.shape
)


# ============================================================
# SEPARATE CLASSES
# ============================================================

non_flow = data[
    data["Label"] == 0
]

flow = data[
    data["Label"] == 1
]


print(
    "\nNon-Flow samples:",
    len(non_flow)
)

print(
    "Flow samples    :",
    len(flow)
)


# ============================================================
# COHEN'S D
# ============================================================

def cohens_d(
    group1,
    group2
):

    mean1 = group1.mean()
    mean2 = group2.mean()

    std1 = group1.std()
    std2 = group2.std()

    n1 = len(group1)
    n2 = len(group2)

    pooled_std = np.sqrt(

        (
            (n1 - 1) * std1 ** 2
            +
            (n2 - 1) * std2 ** 2
        )
        /
        (
            n1 + n2 - 2
        )

    )

    if pooled_std == 0:

        return 0.0

    return (
        (mean1 - mean2)
        /
        pooled_std
    )


# ============================================================
# ANALYSIS
# ============================================================

results = []


for feature in FEATURES:

    non_flow_values = (
        non_flow[feature]
        .dropna()
    )

    flow_values = (
        flow[feature]
        .dropna()
    )


    # --------------------------------------------------------
    # Mann-Whitney U test
    # --------------------------------------------------------

    statistic, p_value = mannwhitneyu(

        non_flow_values,

        flow_values,

        alternative="two-sided"
    )


    # --------------------------------------------------------
    # Effect size
    # --------------------------------------------------------

    effect_size = cohens_d(

        non_flow_values,

        flow_values
    )


    # --------------------------------------------------------
    # Significance
    # --------------------------------------------------------

    if p_value < 0.001:

        significance = "***"

    elif p_value < 0.01:

        significance = "**"

    elif p_value < 0.05:

        significance = "*"

    else:

        significance = "NS"


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append({

        "Feature": feature,

        "NonFlow_Mean":
            non_flow_values.mean(),

        "Flow_Mean":
            flow_values.mean(),

        "NonFlow_Median":
            non_flow_values.median(),

        "Flow_Median":
            flow_values.median(),

        "NonFlow_Std":
            non_flow_values.std(),

        "Flow_Std":
            flow_values.std(),

        "Mann_Whitney_U":
            statistic,

        "P_Value":
            p_value,

        "Cohens_d":
            effect_size,

        "Absolute_Cohens_d":
            abs(effect_size),

        "Significance":
            significance

    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================

# Bonferroni correction for 18 features

results_df["Adjusted_P_Value"] = (
    results_df["P_Value"] * len(FEATURES)
).clip(
    upper=1.0
)


def adjusted_significance(p):

    if p < 0.001:

        return "***"

    elif p < 0.01:

        return "**"

    elif p < 0.05:

        return "*"

    else:

        return "NS"


results_df["Adjusted_Significance"] = (
    results_df["Adjusted_P_Value"]
    .apply(adjusted_significance)
)


# ============================================================
# SORT BY P-VALUE
# ============================================================

results_df = results_df.sort_values(

    by="Adjusted_P_Value",

    ascending=True

).reset_index(
    drop=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("STATISTICAL RESULTS")
print("=" * 70)


display_columns = [

    "Feature",

    "NonFlow_Mean",

    "Flow_Mean",

    "Cohens_d",

    "P_Value",

    "Adjusted_P_Value",

    "Adjusted_Significance"

]


print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        formatters={

            "NonFlow_Mean":
                "{:.4f}".format,

            "Flow_Mean":
                "{:.4f}".format,

            "Cohens_d":
                "{:.4f}".format,

            "P_Value":
                "{:.6e}".format,

            "Adjusted_P_Value":
                "{:.6e}".format

        }
    )
)


# ============================================================
# SIGNIFICANT FEATURES
# ============================================================

significant = results_df[
    results_df["Adjusted_P_Value"] < 0.05
]


print("\n")
print("=" * 70)
print("STATISTICALLY SIGNIFICANT FEATURES")
print("=" * 70)


if len(significant) == 0:

    print(
        "\nNo features remained significant after "
        "Bonferroni correction."
    )

else:

    print(
        "\nNumber of significant features:",
        len(significant)
    )

    print()

    for i, row in enumerate(
        significant.itertuples(),
        start=1
    ):

        print(

            f"{i}. {row.Feature} | "

            f"Adjusted p = "
            f"{row.Adjusted_P_Value:.6e} | "

            f"Cohen's d = "
            f"{row.Cohens_d:.4f}"

        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(

    OUTPUT_FILE,

    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("RESULT 9 COMPLETED")
print("=" * 70)

print(
    "\nStatistical results saved to:"
)

print(
    OUTPUT_FILE
)