import streamlit as st
import pandas as pd
from utils.helpers import load_fairface, bar_counts, representation_ratio

FEATURES = ["gender", "race", "age"]

def app():
    st.header("Dataset Bias Explorer")
    st.write("Explore **representation bias** in the FairFace labels by selecting any feature.")

    df = load_fairface("/mnt/data/fairface_label_train.csv")
    present = [f for f in FEATURES if f in df.columns]
    if not present:
        st.error("No expected columns (gender, race, age) found in the labels file.")
        return

    col = st.selectbox("Choose a feature to explore:", present, index=0)

    fig, counts = bar_counts(df, col, title=f"{col.capitalize()} Distribution", rotate=20)
    st.pyplot(fig)

    ratio, min_g, max_g = representation_ratio(df, col)
    st.markdown(f"**Representation ratio (min/max)** for `{col}`: **{ratio:.2f}**")
    st.caption("Rule of thumb: values closer to 1.0 indicate more balanced representation.")
    if ratio < 0.8:
        st.warning(f"Under-representation detected for `{min_g}` vs `{max_g}` (ratio {ratio:.2f} < 0.80).")
    else:
        st.success("Representation is reasonably balanced by the 80% heuristic.")

    st.markdown("---")
    st.subheader("Group counts")
    st.dataframe(counts.rename("count").to_frame())
