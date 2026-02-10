# TOPSIS-Based Selection of Pretrained Text Classification Models (V2)

## 📌 Problem Statement

The objective of this project is to identify the most suitable pretrained NLP model for **Text Classification** using the **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** multi-criteria decision-making approach.

Instead of selecting a model solely based on accuracy, multiple performance and efficiency parameters are considered simultaneously. This reflects real-world deployment scenarios where computational cost is as important as predictive performance.

---

## 📊 Dataset Description

**Dataset Used:** AG News  

- Multi-class news classification dataset  
- 4 categories: World, Sports, Business, Sci/Tech  
- 500 samples from the test split were used for evaluation  

Models were evaluated in a **zero-shot configuration** (no fine-tuning), directly using Hugging Face pretrained checkpoints.

---

## 🤖 Pretrained Models Evaluated

The following Hugging Face models were compared:

1. BERT (`bert-base-uncased`)  
2. RoBERTa (`roberta-base`)  
3. DistilBERT (`distilbert-base-uncased`)  
4. ALBERT (`albert-base-v2`)  

---

## 📐 Evaluation Criteria

Four criteria were selected to represent both performance and computational efficiency:

| Criterion | Type | Description |
|-----------|------|-------------|
| Accuracy | Benefit | Classification correctness |
| F1 Score | Benefit | Balance between precision and recall |
| Inference Time (ms) | Cost | Prediction latency |
| Model Size (MB) | Cost | Memory footprint |

### Weights Assigned

| Metric | Weight |
|--------|--------|
| Accuracy | 0.3 |
| F1 Score | 0.3 |
| Inference Time | 0.2 |
| Model Size | 0.2 |

(Benefit criteria are maximized, cost criteria are minimized.)

---

## 🔬 Methodology

### Step 1 — Model Evaluation

Each pretrained model was used to predict labels on the AG News dataset.  
For every model, the following were measured:

- Accuracy  
- F1 Score (weighted)  
- Total inference time  
- Model size on disk  

These values form the **decision matrix**.

---

### Step 2 — Decision Matrix Construction

| Model | Accuracy | F1 Score | Inference Time (ms) | Model Size (MB) |
|-------|----------|----------|---------------------|-----------------|
| BERT | 0.222 | 0.1379 | 60407 | 418 |
| RoBERTa | 0.210 | 0.0736 | 51388 | 478 |
| DistilBERT | 0.276 | 0.1947 | 28602 | 256 |
| ALBERT | 0.216 | 0.1242 | 44643 | 46 |

---

### Step 3 — TOPSIS Algorithm

TOPSIS was applied using the following steps:

1. Normalize the decision matrix  
2. Multiply normalized values by weights  
3. Determine ideal best and ideal worst solutions  
4. Calculate Euclidean distance from ideal best and worst  
5. Compute TOPSIS score  
Score = Distance from Worst / (Distance from Best + Distance from Worst)


6. Rank models based on score  

---

## 🏆 Results

### TOPSIS Ranking

| Rank | Model | TOPSIS Score |
|------|-------|--------------|
| 1 | DistilBERT | 0.7308 |
| 2 | ALBERT | 0.6065 |
| 3 | BERT | 0.3309 |
| 4 | RoBERTa | 0.0896 |

---

## 📈 Result Visualization

A bar chart of TOPSIS scores was generated to visually compare models:
![TOPSIS Score Comparison](plots/topsis_bar_chart.png)

The graph clearly shows DistilBERT achieving the highest score, indicating the best overall balance among all criteria.

---

## ✅ Final Conclusion

Although ALBERT has the smallest model size, its lower predictive performance reduced its overall score.  
RoBERTa showed relatively high computational cost with limited accuracy gains.

**DistilBERT achieved the highest TOPSIS score (0.73)** due to:

- Highest accuracy  
- Highest F1 score  
- Fastest inference  
- Moderate model size  

Therefore, **DistilBERT is selected as the optimal pretrained model for text classification** in this study.

---
## 🔮 Future Work

- Fine-tuning models on domain-specific datasets  
- Sensitivity analysis of TOPSIS weights  
- Deployment benchmarking  
- Inclusion of energy consumption metrics  

---

## 🧠 Key Learning

This project demonstrates how **multi-criteria decision making** can be effectively combined with NLP to select models suitable for real-world deployment rather than relying only on accuracy.


