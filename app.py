import streamlit as st
from apps import text_demo, image_demo, challenge_demo
from pages import info

st.set_page_config(
    page_title="BiasLab – AI Bias Demos",
    layout="wide",
    page_icon="",
)

PAGES = {
    "Text Bias Demo": text_demo,
    "Image Bias Demo": image_demo,
    "Bias Challenge": challenge_demo,
    "ℹAbout": info,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to:", list(PAGES.keys()))

page = PAGES[selection]
page.app()
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sentiment_experiment import sentiment_experiment

st.set_page_config(page_title="BiasLab Lite", layout="wide", page_icon="⚖️")

st.title("⚖️ BiasLab Lite – AI Bias Sandbox")
st.write("Upload a dataset or try the built-in biased dataset (Adult Income).")

dataset_choice = st.radio(
    "Choose dataset:",
    ["Built-in Adult Income dataset", "Upload your own CSV"]
)

df = None

if dataset_choice == "Built-in Adult Income dataset":
    with st.spinner("Loading Adult dataset..."):
        adult = fetch_openml(name="adult", version=2, as_frame=True)
        df = adult.frame

        df.replace("?", pd.NA, inplace=True)
        df.dropna(inplace=True)                
        df.reset_index(drop=True, inplace=True)

        st.success("Loaded and cleaned Adult Income dataset ")
        st.write("Shape:", df.shape)
        st.write(df.head())

elif dataset_choice == "📂 Upload your own CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.success("Uploaded dataset ")
        st.write("Shape:", df.shape)
        st.write(df.head())

if df is not None:
    st.subheader(" Run Bias Experiment")

    target = st.selectbox("Select target column (what you want to predict)", df.columns)

    sensitive = st.selectbox("Select sensitive attribute (e.g., sex, race)", df.columns)

    if st.button("Run Experiment"):
        try:
            X = df.drop(columns=[target])
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
            numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

            pre = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
                    ("num", StandardScaler(), numeric),
                ],
                remainder="drop",
            )

            clf = Pipeline([
                ("pre", pre),
                ("model", LogisticRegression(max_iter=500))
            ])

            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            acc = accuracy_score(y_test, preds)

            st.metric("Accuracy", f"{acc:.2f}")

            if sensitive in df.columns:
                st.subheader("Bias Check")
                bias_df = pd.DataFrame({
                    "y_true": y_test,
                    "y_pred": preds,
                    "sensitive": X_test[sensitive].values
                })

                group_acc = bias_df.groupby("sensitive").apply(
                    lambda g: accuracy_score(g["y_true"], g["y_pred"])
                )

                st.write("Accuracy by sensitive group:")
                st.write(group_acc)

        except Exception as e:
            st.error(f"Error during experiment: {e}")
            
sentiment_experiment()
