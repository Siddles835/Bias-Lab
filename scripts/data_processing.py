import pandas as pd

def load_text_dataset(path="data/text_bias.csv"):
    """Load a small text bias dataset."""
    df = pd.read_csv(path)
    return df

def load_image_dataset(path="data/images/"):
    """Return list of image file paths."""
    import os
    images = [os.path.join(path, f) for f in os.listdir(path) if f.endswith((".jpg",".png"))]
    return images

def preprocess_text(text):
    """Lowercase and strip text"""
    return text.lower().strip()

def balance_dataset(df, group_col="Gender", target_col="Job"):
    """Simple reweighting to balance dataset"""
    counts = df.groupby(group_col).size()
    max_count = counts.max()
    df_list = []
    for g in counts.index:
        group_df = df[df[group_col]==g]
        n_repeat = max_count // len(group_df)
        df_list.append(pd.concat([group_df]*n_repeat, ignore_index=True))
    return pd.concat(df_list, ignore_index=True)
