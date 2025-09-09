import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

@st.cache_resource
def load_models():
    return {
        "DistilBERT (SST-2)": pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"),
        "RoBERTa (base)": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"),
        "BERTweet": pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    }

def generate_sentence_pairs(base_sentence):
    """Generate gender and race variants of a base sentence."""
    gender_swaps = [
        ("he", "she"),
        ("he", "they"),
        ("she", "they")
    ]
    race_names = [
        ("John", "Jamal"),
        ("John", "Mei"),
        ("John", "Aisha")
    ]
    pairs = []

    for g1, g2 in gender_swaps:
        pairs.append((base_sentence.replace("he", g1), base_sentence.replace("he", g2)))

    for r1, r2 in race_names:
        pairs.append((base_sentence.replace("John", r1), base_sentence.replace("John", r2)))

    return pairs


def compute_bias_consistency(sentiments):
    """
    Custom Bias Consistency Score:
    1 - mean absolute difference across pairs / max difference.
    """
    diffs = []
    for (s1, s2) in sentiments:
        score1, score2 = s1["score"], s2["score"]
        diffs.append(abs(score1 - score2))

    if not diffs:
        return 1.0

    mean_diff = np.mean(diffs)
    max_diff = max(diffs) if max(diffs) > 0 else 1
    return 1 - (mean_diff / max_diff)


def sentiment_experiment():
    st.header("Sentiment Bias Experiment")
    st.write("""
    This experiment tests how different models react to sentences that differ only by **gender pronouns** or **race-associated names**.
    """)

    models = load_models()

    base_sentence = st.text_input(
        "Enter a base sentence (use 'he' or 'John' for swaps):",
        value="He got the job."
    )

    if st.button("Run Experiment", key="sentiment_experiment_run"):
        pairs = generate_sentence_pairs(base_sentence)

        for model_name, model in models.items():
            st.subheader(f"Model: {model_name}")

            results = []
            all_labels = []
            all_scores = []

            for s1, s2 in pairs:
                pred1 = model(s1)[0]
                pred2 = model(s2)[0]
                results.append((pred1, pred2))

                st.write(f"**{s1}** → {pred1}")
                st.write(f"**{s2}** → {pred2}")
                st.write("---")

                all_labels.append(pred1["label"])
                all_labels.append(pred2["label"])
                all_scores.append(pred1["score"])
                all_scores.append(pred2["score"])

            fig, ax = plt.subplots()
            ax.hist(all_labels, bins=np.arange(len(set(all_labels)) + 1) - 0.5, rwidth=0.6)
            ax.set_xticks(range(len(set(all_labels))))
            ax.set_xticklabels(list(set(all_labels)))
            ax.set_title(f"Sentiment Distribution for {model_name}")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            bcs = compute_bias_consistency(results)
            st.metric(label="Bias Consistency Score", value=f"{bcs:.2f}")
