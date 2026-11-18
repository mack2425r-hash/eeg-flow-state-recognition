import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = r"D:\paper\advanced_training_data.csv"

MODEL_FILE = r"D:\paper\flow_random_forest_balanced.pkl"

RESULT_FILE = r"D:\paper\rf_balanced_results.txt"

IMPORTANCE_FILE = r"D:\paper\rf_balanced_feature_importance.csv"

RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n")
print("=" * 70)
print("      CLASS-BALANCED RANDOM FOREST")
print("      EEG FLOW / NON-FLOW CLASSIFICATION")
print("=" * 70)


data = pd.read_csv(DATA_FILE)

print("\nDataset Shape:", data.shape)


# ============================================================
# 2. CLEAN DATA
# ============================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna()


# ============================================================
# 3. FEATURES AND LABEL
# ============================================================

X = data.drop(
    "Label",
    axis=1
)

y = data["Label"]


print("\nNumber of Features:", X.shape[1])


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\n")
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# 5. CLASS-BALANCED RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1
)


# ============================================================
# 6. HYPERPARAMETER SEARCH
# ============================================================

param_grid = {

    "n_estimators": [
        100,
        200,
        300
    ],

    "max_depth": [
        None,
        10,
        20
    ],

    "min_samples_split": [
        2,
        5
    ],

    "min_samples_leaf": [
        1,
        2
    ],

    "max_features": [
        "sqrt",
        "log2"
    ],

    "criterion": [
        "gini",
        "entropy"
    ]

}


# ============================================================
# 7. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE
)


print("\n")
print("=" * 70)
print("STARTING BALANCED GRID SEARCH")
print("=" * 70)

print(
    "\nParameter combinations:",
    np.prod([
        len(v)
        for v in param_grid.values()
    ])
)

print(
    "Total CV fits:",
    np.prod([
        len(v)
        for v in param_grid.values()
    ]) * 5
)


# ============================================================
# 8. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    scoring="f1",

    cv=cv,

    n_jobs=-1,

    verbose=1,

    return_train_score=False
)


grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 9. BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_


print("\n")
print("=" * 70)
print("BEST BALANCED MODEL")
print("=" * 70)

print("\nBest Parameters:")

print(
    grid_search.best_params_
)

print(
    "\nBest Cross-Validation F1:",
    round(
        grid_search.best_score_,
        4
    )
)


# ============================================================
# 10. TEST SET PREDICTION
# ============================================================

y_pred = best_model.predict(
    X_test
)

y_probability = best_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 11. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)

report = classification_report(

    y_test,

    y_pred,

    target_names=[
        "Non-Flow",
        "Flow"
    ],

    zero_division=0
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL BALANCED TEST PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(cm)

print("\nTrue Non-Flow :", cm[0, 0])

print("False Flow    :", cm[0, 1])

print("False Non-Flow:", cm[1, 0])

print("True Flow     :", cm[1, 1])


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()

print(report)


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        best_model.feature_importances_

})


importance = importance.sort_values(

    by="Importance",

    ascending=False

).reset_index(
    drop=True
)


print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print()

print(
    importance.to_string(
        index=False
    )
)


importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ============================================================
# 16. SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_FILE
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CLASS-BALANCED RANDOM FOREST\n"
    )

    file.write(
        "============================\n\n"
    )

    file.write(
        f"Dataset Samples : {len(data)}\n"
    )

    file.write(
        f"Features        : {X.shape[1]}\n"
    )

    file.write(
        f"Training        : {len(X_train)}\n"
    )

    file.write(
        f"Testing         : {len(X_test)}\n\n"
    )

    file.write(
        "Best Parameters:\n"
    )

    file.write(
        str(
            grid_search.best_params_
        )
    )

    file.write("\n\n")

    file.write(
        f"Best CV F1 : "
        f"{grid_search.best_score_:.4f}\n\n"
    )

    file.write(
        "TEST SET RESULTS\n"
    )

    file.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    file.write(
        f"Precision : {precision:.4f}\n"
    )

    file.write(
        f"Recall    : {recall:.4f}\n"
    )

    file.write(
        f"F1-Score  : {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC   : {roc_auc:.4f}\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\nClassification Report:\n"
    )

    file.write(
        report
    )


# ============================================================
# 18. FINAL
# ============================================================

print("\n")
print("=" * 70)
print("FILES GENERATED")
print("=" * 70)

print(
    "\nBalanced Model:"
)

print(
    MODEL_FILE
)

print(
    "\nResults:"
)

print(
    RESULT_FILE
)

print(
    "\nFeature Importance:"
)

print(
    IMPORTANCE_FILE
)

print("\nBalanced RF experiment completed.")