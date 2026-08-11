import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# HUGGING FACE MODEL
# ============================================================

MODEL_NAME = "anbucode/fake-news-distilbert"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    # Use GPU if available, otherwise CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# PREDICTION
# ============================================================

def predict(
    text,
    tokenizer,
    model,
    device
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # Move inputs to same device as model
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Disable gradients during prediction
    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    return (
        prediction,
        probabilities.cpu().numpy()[0]
    )