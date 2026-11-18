import pandas as pd


FILE = r"D:\paper\dataset\subject1_eeg.csv"


data = pd.read_csv(FILE)


print("\n==============================================")
print("EVENT STRUCTURE CHECK")
print("==============================================")

print("\nTotal samples:", len(data))


print("\nEvent value counts:")
print(
    data["event"].value_counts(dropna=False)
)


print("\nMissing event values:")
print(
    data["event"].isna().sum()
)


print("\nNumber of unique events:")
print(
    data["event"].nunique()
)


print("\nUnique event values:")
print(
    data["event"].unique()
)


print("\nFirst 20 events:")
print(
    data["event"].head(20).to_string(index=False)
)


print("\nLast 20 events:")
print(
    data["event"].tail(20).to_string(index=False)
)


print("\n==============================================")
print("CONSECUTIVE EVENT TRANSITIONS")
print("==============================================")


changes = (
    data["event"]
    .ne(data["event"].shift())
)


transition_rows = data.loc[
    changes,
    ["time", "event"]
]


print(
    transition_rows.to_string(
        index=False
    )
)


print("\nNumber of event segments:")
print(
    len(transition_rows)
)