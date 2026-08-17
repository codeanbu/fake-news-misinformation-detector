print("🔥 LOADED MY NEW EXPLAIN.PY")
import torch
import numpy as np
from lime.lime_text import LimeTextExplainer


explainer = LimeTextExplainer(
    class_names=["Fake", "Real"]
)


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

            probs = torch.softmax(
                outputs.logits,
                dim=1
            )

        predictions.append(
            probs[0].detach().cpu().numpy()
        )

    # IMPORTANT
    # Convert the complete result to a NumPy matrix
    return np.array(predictions, dtype=np.float64)


def explain_prediction(text, tokenizer, model, device):

    def classifier_fn(texts):

        result = predict_proba(
            texts,
            tokenizer,
            model,
            device
        )

        # Force NumPy array
        result = np.asarray(
            result,
            dtype=np.float64
        )

        # Make sure it is 2-dimensional
        if result.ndim == 1:
            result = result.reshape(1, -1)

        return result

    explanation = explainer.explain_instance(
        text,
        classifier_fn,
        labels=(0, 1),
        num_features=10,
        num_samples=500
    )

    return explanation.as_list()