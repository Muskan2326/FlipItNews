
import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="FlipItNews AI",
    page_icon="📰",
    layout="wide",
)

st.title("📰 FlipItNews AI")
st.write("AI-powered news article classification using an LSTM model.")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        MODEL_DIR / "best_rnn.keras"
    )

    tokenizer = tokenizer_from_json(
        (MODEL_DIR / "tokenizer.json").read_text(
            encoding="utf-8"
        )
    )

    classes = json.loads(
        (MODEL_DIR / "label_classes.json").read_text(
            encoding="utf-8"
        )
    )

    config = json.loads(
        (MODEL_DIR / "config.json").read_text(
            encoding="utf-8"
        )
    )

    return model, tokenizer, classes, config


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


model_path = MODEL_DIR / "best_rnn.keras"

if not model_path.exists():
    st.warning(
        "Trained model not found. Run the notebook "
        "and save the model artifacts first."
    )
    st.stop()

model, tokenizer, classes, config = load_artifacts()

article = st.text_area(
    "Enter your news article",
    height=220,
    placeholder="Paste the headline and article text here...",
)
max_length = 200 

# article = user-entered news text
sequence = tokenizer.texts_to_sequences([article])

padded_sequence = pad_sequences(
    sequence,
    maxlen=max_length,
    padding="post",
    truncating="post"
)


if st.button("Classify News", type="primary"):
    if not article.strip():
        st.error("Please enter an article first.")
    else:
        cleaned = preprocess(article)

        sequence = tokenizer.texts_to_sequences([cleaned])

        padded = pad_sequences(
            sequence,
            maxlen=int(config["max_len"]),
            padding="post",
            truncating="post",
        )

        predictions = model.predict(padded_sequence)

        probabilities = predictions[0]

        # Debugging: inspect the model's predictions

        order = np.argsort(probabilities)[::-1]

        st.subheader("Prediction diagnostics")

        for i in order[:3]:
            st.write(
                f"{classes[i].title()}: "
                f"{probabilities[i] * 100:.2f}%"
            )

        st.write("Input after preprocessing:")
        st.code(cleaned[:1000])


        best_index = int(np.argmax(probabilities))

        st.success(
            f"Predicted category: "
            f"{classes[best_index].title()}"
        )

        st.metric(
            "Model confidence",
            f"{probabilities[best_index] * 100:.2f}%"
        )

        st.subheader("Category probabilities")

        probability_data = {
            classes[i]: float(probabilities[i])
            for i in range(len(classes))
        }

        st.bar_chart(probability_data)

        st.caption(
            "Confidence is a model estimate, not a guarantee "
            "that the prediction is correct."
        )
