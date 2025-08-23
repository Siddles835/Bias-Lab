import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def app():
    st.header("Bias Challenge")
    st.write("Try to **fix bias** in a simulated dataset yourself!")

    st.subheader("Step 1: Current Accuracy by Group")
    groups = ["Male-Light", "Male-Dark", "Female-Light", "Female-Dark"]
    acc = np.array([0.95, 0.75, 0.85, 0.65])

    fig, ax = plt.subplots()
    ax.bar(groups, acc, color=["#4CAF50", "#F44336", "#2196F3", "#FF9800"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Before Mitigation")
    st.pyplot(fig)

    st.subheader("Step 2: Choose a Mitigation Strategy")
    strategy = st.radio(
        "Select bias mitigation method:",
        ["Re-weighting", "Data Augmentation", "Adversarial Training"]
    )

    if st.button("Apply Fix"):
        if strategy == "Re-weighting":
            fixed_acc = [0.88, 0.85, 0.87, 0.84]
        elif strategy == "Data Augmentation":
            fixed_acc = [0.90, 0.82, 0.89, 0.81]
        else:  
            fixed_acc = [0.87, 0.86, 0.86, 0.85]

        fig2, ax2 = plt.subplots()
        ax2.bar(groups, fixed_acc, color="skyblue")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"After Mitigation: {strategy}")
        st.pyplot(fig2)

        st.success("🎉 Bias reduced successfully (simulation)!")