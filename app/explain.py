import torch
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(
    class_names=["Fake", "Real"]
)

def predict_proba(texts, tokenizer, model, device):

    predictions = []

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)

        predictions.append(probs.cpu().numpy()[0])

    return predictions


def explain_prediction(text, tokenizer, model, device):

    explanation = explainer.explain_instance(
        text,
        lambda x: predict_proba(x, tokenizer, model, device),
        num_features=10
    )

    return explanation.as_list()