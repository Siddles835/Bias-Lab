import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def app():
    st.header("📝 Text Bias Demo")
    st.write("Explore how AI models may show **gender or racial bias** in text analysis.")

    user_input = st.text_area("Enter text here:", "The doctor said he will help.")

    classifier = pipeline("sentiment-analysis")

    gender_words = {"he": "male", "she": "female", "him": "male", "her": "female"}
    racial_words = {"John": "white", "Ahmed": "middle_eastern", "Aisha": "middle_eastern"}

    if st.button("Analyze"):
        result = classifier(user_input)
        st.subheader("Sentiment Analysis Result:")
        st.json(result)

        words = user_input.lower().split()
        gender_detected = [gender_words[w] for w in words if w in gender_words]
        racial_detected = [racial_words[w.capitalize()] for w in words if w.capitalize() in racial_words]

        if gender_detected:
            st.warning(f"Gender-associated words detected: {', '.join(gender_detected)}")
        if racial_detected:
            st.warning(f"Race-associated words detected: {', '.join(racial_detected)}")
        if not gender_detected and not racial_detected:
            st.success("No explicit gender or racial words detected.")

        st.subheader("Bias Score Simulation")
        data = pd.DataFrame({
            "Category": ["Male", "Female", "Other"],
            "Sentiment Score": [
                0.8 if "male" in gender_detected else 0.5,
                0.6 if "female" in gender_detected else 0.5,
                0.5
            ]
        })
        fig, ax = plt.subplots()
        sns.barplot(x="Category", y="Sentiment Score", data=data, ax=ax, palette="coolwarm")
        st.pyplot(fig)

    st.info("This demo simulates bias detection in text using toy logic.")