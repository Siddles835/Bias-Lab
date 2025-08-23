import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def app():
    st.header("Image Bias Demo")
    st.write("Simulated demo of **bias in facial recognition systems** (no real dataset required).")

    st.subheader("Step 1: Upload a Face (optional)")
    uploaded_file = st.file_uploader("Upload a face image (optional, any picture)", type=["jpg", "jpeg", "png"])

    st.subheader("Step 2: Simulated Classifier")
    if st.button("Run Biasy Classifier"):
        st.warning("Simulated model predicts better accuracy for lighter-skinned faces and males (example of bias).")

        groups = ["Male-Light", "Male-Dark", "Female-Light", "Female-Dark"]
        acc = [0.95, 0.75, 0.85, 0.65] 

        fig, ax = plt.subplots()
        ax.bar(groups, acc, color=["#4CAF50", "#F44336", "#2196F3", "#FF9800"])
        ax.set_ylabel("Accuracy")
        ax.set_title("Simulated Model Accuracy by Demographic")
        st.pyplot(fig)

    st.info("In reality, models trained on unbalanced datasets show **accuracy gaps across demographics**.")