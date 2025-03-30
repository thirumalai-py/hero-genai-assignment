import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("airquality.csv")

# Replace -200 with NaN
df = df.replace(-200, np.nan)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values per Column:\n", missing_values)

# Get the percentage of missing values
total_cells = np.prod(df.shape)
missing_cells = df.isnull().sum().sum()
percentage_missing = (missing_cells / total_cells) * 100
print(f"\nPercentage of missing values in the dataset: {percentage_missing:.2f}%")

# Detailed missing value analysis
print("\nMissing Value Details:")
for column in df.columns:
    missing_count = df[column].isnull().sum()
    if missing_count > 0:
        missing_percentage = (missing_count / len(df)) * 100
        print(f"{column}: {missing_count} missing values ({missing_percentage:.2f}%)")

# Imputation strategies
# 1. Forward fill
df_forward_filled = df.ffill()

# 2. Backward fill
df_backward_filled = df.bfill()

# 3. Mean imputation for numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# Optional: Save cleaned dataset
df.to_csv("airquality_cleaned.csv", index=False)

print("\nCleaning complete. Cleaned dataset saved as 'airquality_cleaned.csv'")