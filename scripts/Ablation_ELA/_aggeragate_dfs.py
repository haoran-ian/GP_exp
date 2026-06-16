import pandas as pd

df1 = pd.read_csv(
    "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela_cached.csv")
df2 = pd.read_csv(
    "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv")
only_in_df1 = set(df1.columns) - set(df2.columns)

# df2 有但 df1 没有的列
only_in_df2 = set(df2.columns) - set(df1.columns)

print("Only in df1:", only_in_df1)
print("Only in df2:", only_in_df2)
common_cols = df1.columns.intersection(df2.columns)

print(common_cols)
common_cols = df1.columns.intersection(df2.columns)

df_merged = pd.concat(
    [df1[common_cols], df2[common_cols]],
    axis=0,
    ignore_index=True
)
df_merged.to_csv(
    "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv", index=False)
