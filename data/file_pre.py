import pandas as pd

file_path = "./data/File1.csv"

data = pd.read_csv(file_path)
species = [
    "Picea abies",
    "Fagus sylvatica",
    "Pinus sylvestris",
    "Quercus robur",
    "Betula pendula",
    "Quercus petraea",
    "Fraxinus excelsior",
    "Acer pseudoplatanus",
    "Sorbus aucuparia",
    "Carpinus betulus",
]
"""
contry = []


filtered_df = data[data["SPECIES NAME"].isin(species)]
filtered_df.to_csv("EU_10species.csv", index=False)
"""

filtered_df = data[data["SPECIES NAME"].isin(species)]
unique_df = filtered_df.drop_duplicates(subset=["X", "Y"], keep=False)
print(unique_df)
unique_df.to_csv("10_strict_norepeat.csv", index=False)


"""
duplicates = data.duplicated(subset=["X", "Y"], keep="first")
duplicate_rows = data[duplicates]
grouped = duplicate_rows.groupby(["X", "Y"])
for (x, y), group in grouped:
    print(f"Group for X={x} and Y={y}:")
    print(group)
    print("\n")
"""
