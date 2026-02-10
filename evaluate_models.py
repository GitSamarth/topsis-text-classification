import time
import os
import torch
import pandas as pd
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

# -------------------------------
# Configuration
# -------------------------------
MODELS = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "DistilBERT": "distilbert-base-uncased",
    "ALBERT": "albert-base-v2"
}

NUM_SAMPLES = 500   # keep small for faster execution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Helper: model size
# -------------------------------
def get_model_size(model_dir):
    total_size = 0
    for dirpath, _, filenames in os.walk(model_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # MB

# -------------------------------
# Load dataset
# -------------------------------
dataset = load_dataset("ag_news", split="test[:{}]".format(NUM_SAMPLES))
texts = dataset["text"]
labels = dataset["label"]

results = []

# -------------------------------
# Evaluate each model
# -------------------------------
for model_name, model_id in MODELS.items():
    print(f"\nEvaluating {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=4)
    model.to(DEVICE)
    model.eval()

    predictions = []

    start_time = time.time()

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(DEVICE)

            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)

    end_time = time.time()

    # Metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    inference_time = (end_time - start_time) * 1000  # ms

    # Save model locally to measure size
    save_path = f"models/{model_name}"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    model_size = get_model_size(save_path)

    results.append([
        model_name,
        accuracy,
        f1,
        inference_time,
        model_size
    ])

# -------------------------------
# Save Decision Matrix
# -------------------------------
df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "F1_Score", "Inference_Time_ms", "Model_Size_MB"]
)

os.makedirs("results", exist_ok=True)
df.to_csv("results/decision_matrix.csv", index=False)

print("\nDecision matrix saved to results/decision_matrix.csv")
print(df)

