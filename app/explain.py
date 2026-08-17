import torch
import numpy as np
from lime.lime_text import LimeTextExplainer


# ============================================================
# LIME EXPLAINER
# ============================================================

explainer = LimeTextExplainer(
    class_names=["Fake", "Real"]
)


# ============================================================
# PREDICTION FUNCTION FOR LIME
# ============================================================

def predict_proba(texts, tokenizer, model, device):

    model.eval()

    predictions = []

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

        prediction = (
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
        )

        predictions.append(prediction)

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )

    # LIME requires:
    # (number_of_texts, number_of_classes)

    if predictions.ndim != 2:
        predictions = predictions.reshape(-1, 2)

    return predictions


# ============================================================
# EXPLAIN PREDICTION
# ============================================================

def explain_prediction(
    text,
    tokenizer,
    model,
    device
):

    def classifier_fn(texts):

        result = predict_proba(
            texts,
            tokenizer,
            model,
            device
        )

        result = np.asarray(
            result,
            dtype=np.float64
        )

        if result.ndim != 2:
            result = result.reshape(-1, 2)

        if result.shape[1] != 2:
            raise ValueError(
                "Model must return probabilities "
                "for exactly 2 classes."
            )

        return result

    # ========================================================
    # LIME
    # ========================================================

    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=classifier_fn,
        labels=[0, 1],
        num_features=10,
        num_samples=100
    )

    return explanation.as_list()