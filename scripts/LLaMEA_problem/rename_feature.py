import pandas as pd

cols_to_drop = ["disp.costs_runtime", "ela_distr.costs_runtime",
                "ela_level.costs_runtime", "ela_meta.costs_runtime",
                "ic.costs_runtime", "nbc.costs_runtime", "pca.costs_runtime",
                "y_max", "y_mean", "y_min", "y_std"]
df = pd.read_csv("data/MABBOB/mabbob_selected_ela_raw.csv")

def rename_col(col):
    # 只处理包含多个 "." 的列名
    if col.count(".") > 1:
        return col.split(".", 1)[1]  # 删除第一个 "." 及其之前的字符
    return col

df = df.rename(columns=rename_col)
df = df.drop(columns=cols_to_drop, errors="ignore")

df.to_csv("data/MABBOB/mabbob_selected_ela.csv", index=False)