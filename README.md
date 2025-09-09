# Exposing and Fixing AI Bias: An Interactive Journey Through Text and Images

*By Sidhaanth Kapoor*

---

## Why This Project Exists

Artificial Intelligence is shaping the world around us — from hiring decisions, to loan approvals, to how we interact online. But AI systems are not neutral. They inherit the biases of the data they are trained on, and sometimes even amplify those biases.  

The stakes are real. Imagine a facial recognition system misidentifying people of color at alarmingly higher rates, or a sentiment analysis tool consistently rating sentences with female pronouns more negatively than identical sentences with male pronouns. These are not hypothetical scenarios — they’ve already been observed in the wild.  

This project is my attempt to shine a light on the problem of **algorithmic bias** — not just by writing about it, but by **building interactive experiments** that anyone can try for themselves.  

My vision:  
- **Make bias visible** through demos.  
- **Educate users** on how bias emerges.  
- **Test mitigation methods** with real models.  
- **Propose new ways** of measuring fairness.  

Because only when the problem is transparent, can we demand better solutions.  

---

## The Bias Lab App

I built the **Bias Lab** as a Streamlit app where users can **test AI bias live**. It currently has two major sections:

1. **Text Sentiment Analysis**  
2. **Image Classification Bias**

Both are designed to be **hands-on and interactive**, because reading about bias is one thing — but *seeing bias in action* hits differently.

---

## Part 1: Sentiment Analysis Bias

### What It Does
Users type in any text, and a model predicts whether it’s *positive*, *negative*, or *neutral*.  

At first, this feels harmless. But what if we slightly change the text?  

- “He likes ice cream” → Positive  
- “She likes ice cream” → Neutral  
- “They like ice cream” → Negative  

The meaning is the same. Only the pronoun changed. Yet the model shifts its sentiment output. That’s **bias in action**.

---

### Sentiment Experiment: Bias by Design

To push this further, I added an **experiment mode**.  

- The app automatically generates sentence **pairs** that differ only by **gender pronoun** or **race-associated names**.  
- Each pair is fed into multiple sentiment models.  
- Predictions are compared, and **bias distributions** are visualized with histograms.  

For example:  

- “John got the job.” → 95% Positive  
- “Jamal got the job.” → 67% Positive  

The words are identical except for the name. But the model shows a measurable bias.  

---

### Visualizing Bias

The app plots:  
- **Histograms** of sentiment for male vs. female pronouns.  
- **Side-by-side distributions** for names associated with different racial groups.  
- **Model comparison charts**, showing which models are more consistent.  

This lets users *see* how models diverge in fairness.  

---

### A Custom Metric: Bias Consistency Score

To quantify bias, I designed a **Bias Consistency Score (BCS)**:  

\[
\text{BCS} = 1 - \frac{\text{Mean Absolute Sentiment Difference across pairs}}{\text{Max Difference}}
\]  

- **1.0** → Perfectly consistent (no bias detected).  
- **0.0** → Completely inconsistent (highly biased).  

This metric helps compare models objectively:  

| Model        | Bias Consistency Score |
|--------------|------------------------|
| DistilBERT   | 0.72                   |
| RoBERTa      | 0.81                   |
| BERTweet     | 0.64                   |

While no model is bias-free, some are more reliable than others.  

---

### Bias Mitigation in Text

The app also explores methods to **reduce bias**:  

1. **Data Augmentation**  
   - Add gender-swapped or race-swapped sentences to the training set.  
   - Strength: Simple, scalable.  
   - Weakness: Doesn’t fully address deeper correlations.  

2. **Re-weighting**  
   - Give more weight to underrepresented examples.  
   - Strength: Targets imbalance.  
   - Weakness: Can overfit rare cases.  

3. **Adversarial Debiasing**  
   - Train a secondary model to remove sensitive features (like gender).  
   - Strength: Strong in theory.  
   - Weakness: Computationally expensive.  

The takeaway: there’s no “silver bullet.” The right strategy depends on context.  

---

## Part 2: Image Classification Bias

### What It Does
In the image demo, users can upload a photo and see how the model classifies it — gender, age, or expression.  

### Bias in Action
Studies show that facial recognition systems often misclassify darker-skinned women at much higher rates than lighter-skinned men. My app lets users **test this live** by uploading different images.  

### Interactive Experiment
Users can:  
- Upload faces of diverse backgrounds.  
- Compare predictions (confidence scores, categories).  
- Notice disparities across groups.  

This interactivity empowers users to be **bias auditors themselves**.  

---

## How Bias is Mitigated in Images

Mitigation techniques include:  
- **Data Balancing**: Ensuring equal representation of skin tones, genders, and age groups.  
- **Re-weighting**: Adjusting training loss to emphasize underrepresented demographics.  
- **Domain Adaptation**: Training models that generalize across subpopulations.  

Each comes with trade-offs, and my app highlights these in plain English.  

---

## Why This Matters

This isn’t just a tech demo. It’s about **accountability**.  

- If an AI tool rates resumes with female names lower than male names, it impacts careers.  
- If a sentiment model assigns negativity to tweets with African American vernacular, it impacts online speech.  
- If facial recognition misidentifies people of color, it impacts civil liberties.  

Bias isn’t abstract. It affects real lives.  

---

## What Makes My Project Unique

1. **Hands-On Interactivity** → Not just research papers, but demos anyone can try.  
2. **Multiple Models** → Compare fairness across architectures.  
3. **Custom Metric** → A new way to measure bias consistency.  
4. **Transparency + Education** → Tooltips and explanations for beginners.  

This is not just about showing bias. It’s about *teaching users* how to understand, measure, and fix it.  

---

## Lessons Learned

- **Bias is everywhere**: even in models hailed as “state-of-the-art.”  
- **Mitigation is complex**: every method has strengths and weaknesses.  
- **Transparency is power**: giving users tools to explore bias turns them into critical thinkers.  

---

## What’s Next

- Expand the image demo with more tasks (e.g., emotion recognition, occupation inference).  
- Build datasets specifically designed to **stress-test fairness**.  
- Publish my Bias Consistency Metric and make it available to other researchers.  

---

## The Big Picture

This project is just the beginning. Bias in AI is one of the defining challenges of our generation. If we can expose it, understand it, and mitigate it, we can build technology that serves everyone fairly.  

That’s the future I want to help create.  

---

## Try It Yourself

[Launch the Bias Lab App](https://bias-lab-nraqsmshhttfkg4njefsvq.streamlit.app/)  

---
