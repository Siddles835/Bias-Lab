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
