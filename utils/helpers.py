import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, List, Tuple

# --------- UI ----------
def app_header():
    st.title("AI Bias Lab")
    st.caption("Explore, visualize, and mitigate bias across text and image datasets.")

@st.cache_data(show_spinner=False)
def load_fairface(path: str) -> pd.DataFrame:
    """
    Load FairFace labels CSV. Expected columns (typical): file, age, gender, race, etc.
    We normalize column names and types as needed.
    """
    df = pd.read_csv(path)
    colmap = {c.lower(): c for c in df.columns}
    for k in list(colmap.keys()):
        if k in ["gender", "race", "age", "file"]:
            continue
    for std in ["gender", "race", "age", "file"]:
        if std not in df.columns and std in colmap:
            df.rename(columns={colmap[std]: std}, inplace=True)
    keep_cols = [c for c in ["file", "age", "gender", "race"] if c in df.columns]
    if keep_cols:
        df = df.dropna(subset=keep_cols)
    return df.reset_index(drop=True)

def bar_counts(df: pd.DataFrame, col: str, title: str = "", rotate: int = 0):
    counts = df[col].value_counts(dropna=False).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(title or f"{col} distribution")
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    plt.tight_layout()
    return fig, counts

def rate_barplot(series_dict: Dict[str, float], title: str, ylabel: str):
    items = sorted(series_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots()
    sns.barplot(x=labels, y=vals, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig

def selection_rates(labels: pd.Series, group: pd.Series) -> Dict[str, float]:
    """
    labels: binary 0/1 outcome (e.g., selected=1)
    group: protected attribute (e.g., gender)
    Returns selection rate per group: P(Y=1 | group)
    """
    rates = {}
    for g, sub in labels.groupby(group):
        mask = (group == g)
        if mask.sum() == 0:
            rates[g] = 0.0
        else:
            rates[g] = float(labels[mask].mean())
    return rates

def disparate_impact(labels: pd.Series, group: pd.Series) -> Tuple[float, str, str]:
    """
    DI = min_group_rate / max_group_rate (the 80% rule expects >= 0.8)
    Returns (DI, min_group, max_group)
    """
    rates = selection_rates(labels, group)
    if not rates:
        return 1.0, "", ""
    max_g = max(rates, key=rates.get)
    min_g = min(rates, key=rates.get)
    max_r = rates[max_g] if rates[max_g] > 0 else 1e-9
    di = rates[min_g] / max_r
    return float(di), str(min_g), str(max_g)

def representation_ratio(df: pd.DataFrame, group_col: str) -> Tuple[float, str, str]:
    """
    Representation parity: min_group_prop / max_group_prop
    """
    props = df[group_col].value_counts(normalize=True)
    if props.empty:
        return 1.0, "", ""
    max_g = props.idxmax()
    min_g = props.idxmin()
    ratio = float(props[min_g] / (props[max_g] if props[max_g] > 0 else 1e-9))
    return ratio, str(min_g), str(max_g)

@st.cache_data(show_spinner=False)
def synthesize_outcome(df: pd.DataFrame, protected_col: str,
                       base_rate: float = 0.35,
                       group_multipliers: Dict[str, float] = None,
                       seed: int = 42) -> pd.Series:
    """
    Create a biased binary outcome Y ~ Bernoulli(p), where p depends on protected group.
    group_multipliers: e.g., {"Male": 1.0, "Female": 0.7, "Black": 0.6} (applied only to groups present)
    """
    rng = np.random.default_rng(seed)
    groups = df[protected_col].astype(str)
    if group_multipliers is None:
        group_multipliers = {g: 1.0 for g in groups.unique()}
    ps = groups.map(lambda g: np.clip(base_rate * group_multipliers.get(g, 1.0), 0.01, 0.99))
    y = rng.binomial(1, ps)
    return pd.Series(y, index=df.index, name="selected")

def apply_reweighting(df: pd.DataFrame, protected_col: str, weights: Dict[str, float]) -> pd.Series:
    """
    Produce sample weights aligned to chosen per-group weights (normalized).
    """
    w = df[protected_col].astype(str).map(lambda g: weights.get(g, 1.0)).astype(float)
    w = w / (w.mean() if w.mean() > 0 else 1.0)
    return w

def score_fairness(di: float) -> Tuple[str, str]:
    """
    Turn a disparate impact score into a quick grade + hint.
    """
    if di >= 0.95:
        return "A", "Excellent parity. DI ≥ 0.95."
    if di >= 0.8:
        return "B", "Meets the 80% rule (DI ≥ 0.8)."
    if di >= 0.65:
        return "C", "Some improvement needed. Try stronger mitigation."
    return "D", "Biased outcome. Aim for DI ≥ 0.8 via stronger mitigation."
