import pandas as pd
import os

# Load the raw CSV
df = pd.read_csv("data/wine_dataset.csv")  # <-- ✅ correct path

# Optional: clean, drop duplicates, handle nulls
df.drop_duplicates(inplace=True)

# Save to Parquet format
os.makedirs("data", exist_ok=True)
df.to_parquet("data/cleaned_wine.parquet", index=False)
print("✅ Saved data/cleaned_wine.parquet")
