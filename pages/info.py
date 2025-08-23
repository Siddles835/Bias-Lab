import streamlit as st

def app():
    st.header("About BiasLab")
    st.write(
        """
        **BiasLab** is an interactive sandbox to explore how **bias shows up in AI systems**
        and what we can do to fix it.  

        Features:
        - **Text Bias Demo** → Test sentences for bias in sentiment analysis.
        - **Image Bias Demo** → Explore demographic bias in face classifiers.
        - **Bias Challenge** → Try fairness mitigation strategies.

        This project uses **toy examples** and **simulated data** (no external datasets)
        so that you can explore concepts instantly without setup.
        """
    )
    st.info("Created by a high schooler passionate about ethical AI 🚀")
