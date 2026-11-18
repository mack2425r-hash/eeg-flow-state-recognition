import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

INPUT_FILE = r"D:\paper\rf_balanced_feature_importance.csv"

OUTPUT_FILE = r"D:\paper\feature_importance_plot.png"


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

data = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# SORT FEATURES
# ============================================================

data = data.sort_values(
    by="Importance",
    ascending=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("RESULT 7 - FEATURE IMPORTANCE")
print("==============================================\n")

print(
    data.sort_values(
        by="Importance",
        ascending=False
    ).to_string(
        index=False
    )
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(9, 7)
)


plt.barh(
    data["Feature"],
    data["Importance"]
)


plt.xlabel(
    "Random Forest Feature Importance"
)

plt.ylabel(
    "EEG Feature"
)

plt.title(
    "Feature Importance for EEG Flow State Classification"
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# TOP 5 FEATURES
# ============================================================

top5 = data.sort_values(
    by="Importance",
    ascending=False
).head(5)


print("\n==============================================")
print("TOP 5 FEATURES")
print("==============================================\n")


for i, row in enumerate(
    top5.itertuples(),
    start=1
):

    print(
        f"{i}. {row.Feature} "
        f"= {row.Importance:.6f}"
    )


print("\n==============================================")
print("RESULT 7 COMPLETED")
print("==============================================")

print(
    "\nFigure saved to:"
)

print(
    OUTPUT_FILE
)