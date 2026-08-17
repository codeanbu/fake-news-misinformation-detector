import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from predictor import load_model, predict
from explain import explain_prediction
from url_reader import extract_article


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Fake News & Misinformation Detector",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probs" not in st.session_state:
    st.session_state.probs = None

if "analyzed_news" not in st.session_state:
    st.session_state.analyzed_news = ""

if "explanation" not in st.session_state:
    st.session_state.explanation = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📰 Fake News Detector")

    st.markdown("---")

    st.subheader("🤖 Model")
    st.success("DistilBERT Transformer")

    st.subheader("📊 Dataset")
    st.info("44,889 News Articles")

    st.subheader("🏷 Classes")
    st.write("🔴 Fake")
    st.write("🟢 Real")

    st.markdown("---")

    st.subheader("⚙️ Technology")

    st.write("• PyTorch")
    st.write("• Hugging Face")
    st.write("• Streamlit")
    st.write("• Plotly")
    st.write("• LIME")

    st.markdown("---")

    st.caption("🚀 Developed by Anbuselvan")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def initialize():
    return load_model()


tokenizer, model, device = initialize()


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "📰 AI-Powered Fake News & Misinformation Detector"
)

st.markdown(
    """
    Detect whether a news article is **Fake** or **Real**
    using a **fine-tuned DistilBERT Transformer** model.
    """
)

st.markdown("---")


# ============================================================
# NEWS INPUT
# ============================================================

news = st.text_area(
    "📝 Paste News Article",
    height=250,
    placeholder="Paste your news article here..."
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown("### 📂 Or Upload a News File")

uploaded_file = st.file_uploader(
    "Choose a TXT or CSV file",
    type=["txt", "csv"]
)

batch_df = None


if uploaded_file is not None:

    # ========================================================
    # TXT FILE
    # ========================================================

    if uploaded_file.name.lower().endswith(".txt"):

        news = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

        st.success(
            "✅ TXT file loaded successfully!"
        )

        st.text_area(
            "Uploaded Article",
            news,
            height=250
        )

    # ========================================================
    # CSV FILE
    # ========================================================

    elif uploaded_file.name.lower().endswith(".csv"):

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                "✅ CSV loaded successfully!"
            )

            st.write("### Preview")

            st.dataframe(
                batch_df.head(),
                use_container_width=True
            )

            st.info(
                "Click 'Analyze News' to classify "
                "every article."
            )

        except Exception as e:

            st.error(
                f"❌ Could not read CSV: {e}"
            )


# ============================================================
# NEWS URL
# ============================================================

st.markdown("### 🌐 Or Analyze a News URL")

news_url = st.text_input(
    "Paste a news article URL"
)


if news_url:

    try:

        article = extract_article(
            news_url
        )

        news = article["text"]

        st.success(
            "✅ Article extracted successfully!"
        )

        st.subheader(
            "Extracted Article"
        )

        st.text_area(
            "Article Content",
            news,
            height=300
        )

    except Exception as e:

        st.error(
            f"❌ Could not extract article: {e}"
        )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_clicked = st.button(
    "🔍 Analyze News",
    use_container_width=True
)


if analyze_clicked:

    # ========================================================
    # BATCH CSV PREDICTION
    # ========================================================

    if batch_df is not None:

        if "text" not in batch_df.columns:

            st.error(
                "CSV must contain a column named 'text'."
            )

        else:

            predictions = []
            confidences = []

            total_articles = len(
                batch_df
            )

            if total_articles == 0:

                st.warning(
                    "⚠ The CSV file contains no articles."
                )

            else:

                progress = st.progress(
                    0.0
                )

                for i, article in enumerate(
                    batch_df["text"]
                ):

                    prediction, probs = predict(
                        str(article),
                        tokenizer,
                        model,
                        device
                    )

                    confidence = (
                        float(
                            probs[prediction]
                        )
                        * 100.0
                    )

                    predictions.append(
                        "Fake"
                        if prediction == 0
                        else "Real"
                    )

                    confidences.append(
                        round(
                            confidence,
                            2
                        )
                    )

                    progress.progress(
                        float(
                            (i + 1)
                            / total_articles
                        )
                    )

                batch_df["Prediction"] = (
                    predictions
                )

                batch_df["Confidence"] = (
                    confidences
                )

                st.success(
                    "✅ Batch prediction completed!"
                )

                st.dataframe(
                    batch_df,
                    use_container_width=True
                )

                csv = batch_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

    # ========================================================
    # SINGLE ARTICLE
    # ========================================================

    elif not news or not news.strip():

        st.warning(
            "⚠ Please enter a news article."
        )

    else:

        try:

            # ====================================================
            # MODEL PREDICTION
            # ====================================================

            prediction, probs = predict(
                news,
                tokenizer,
                model,
                device
            )

            # ====================================================
            # SAVE RESULT IN SESSION STATE
            # ====================================================

            st.session_state.prediction = (
                int(prediction)
            )

            st.session_state.probs = (
                probs
            )

            st.session_state.analyzed_news = (
                news
            )

            st.session_state.prediction_done = (
                True
            )

            # Clear previous explanation
            st.session_state.explanation = None

        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


# ============================================================
# DISPLAY SINGLE ARTICLE RESULT
# ============================================================

if st.session_state.prediction_done:

    prediction = (
        st.session_state.prediction
    )

    probs = (
        st.session_state.probs
    )

    analyzed_news = (
        st.session_state.analyzed_news
    )

    # ========================================================
    # CONVERT NUMPY VALUES TO PYTHON FLOAT
    # ========================================================

    fake_probability = float(
        probs[0]
    )

    real_probability = float(
        probs[1]
    )

    confidence = (
        float(
            probs[prediction]
        )
        * 100.0
    )

    st.markdown("---")

    # ========================================================
    # PREDICTION
    # ========================================================

    st.subheader(
        "📌 Prediction"
    )

    if prediction == 0:

        st.error(
            "🔴 Fake News Detected"
        )

    else:

        st.success(
            "🟢 Real News"
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    st.subheader(
        "🎯 Confidence Score"
    )

    progress_value = float(
        max(
            0.0,
            min(
                1.0,
                confidence / 100.0
            )
        )
    )

    st.progress(
        progress_value
    )

    st.write(
        f"## {confidence:.2f}% Confidence"
    )

    # ========================================================
    # PROBABILITY CARDS
    # ========================================================

    st.subheader(
        "📊 Prediction Probabilities"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🔴 Fake",
            f"{fake_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "🟢 Real",
            f"{real_probability * 100:.2f}%"
        )

    # ========================================================
    # PLOTLY CHART
    # ========================================================

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[
                "Fake",
                "Real"
            ],
            y=[
                fake_probability * 100.0,
                real_probability * 100.0
            ],
            text=[
                f"{fake_probability * 100:.2f}%",
                f"{real_probability * 100:.2f}%"
            ],
            textposition="outside"
        )
    )

    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Class",
        yaxis_title="Probability (%)",
        yaxis=dict(
            range=[0, 100]
        ),
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # ARTICLE STATISTICS
    # ========================================================

    st.subheader(
        "📝 Article Statistics"
    )

    st.write(
        f"**Words:** {len(analyzed_news.split())}"
    )

    st.write(
        f"**Characters:** {len(analyzed_news)}"
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "ℹ️ Model Information"
    ):

        st.write(
            "### Model Details"
        )

        st.write(
            "**Model:** DistilBERT"
        )

        st.write(
            "**Framework:** "
            "Hugging Face Transformers"
        )

        st.write(
            "**Backend:** PyTorch"
        )

        st.write(
            "**Dataset Size:** "
            "44,889 News Articles"
        )

        st.write(
            "**Task:** Fake News Classification"
        )

        st.write(
            "**Classes:** Fake / Real"
        )

        st.write(
            f"**Device:** {device}"
        )

    # ========================================================
    # RAW PROBABILITIES
    # ========================================================

    st.subheader(
        "📋 Raw Probabilities"
    )

    st.json(
        {
            "Fake": round(
                fake_probability,
                6
            ),
            "Real": round(
                real_probability,
                6
            )
        }
    )


# ============================================================
# LIME EXPLANATION
# ============================================================

if st.session_state.prediction_done:

    st.markdown("---")

    st.subheader(
        "🧠 Explainable AI"
    )

    st.write(
        "LIME identifies the words that influenced "
        "the model's prediction."
    )

    explain_clicked = st.button(
        "🧠 Explain Prediction",
        use_container_width=True
    )

    if explain_clicked:

        with st.spinner(
            "🧠 Generating explanation... "
            "This may take a few seconds."
        ):

            try:

                explanation = explain_prediction(
                    st.session_state.analyzed_news,
                    tokenizer,
                    model,
                    device
                )

                # Save explanation
                st.session_state.explanation = (
                    explanation
                )

            except Exception as e:

                st.error(
                    f"❌ Explanation failed: {e}"
                )


# ============================================================
# DISPLAY LIME EXPLANATION
# ============================================================

if st.session_state.explanation is not None:

    st.success(
        "✅ Explanation generated successfully!"
    )

    st.subheader(
        "🧠 Top Influential Words"
    )

    for word, weight in (
        st.session_state.explanation
    ):

        weight = float(weight)

        if weight > 0:

            st.success(
                f"✅ {word} ({weight:.3f})"
            )

        else:

            st.error(
                f"❌ {word} ({weight:.3f})"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🚀 AI-Powered Fake News & Misinformation "
    "Detection using DistilBERT | "
    "Developed by Anbuselvan"
)