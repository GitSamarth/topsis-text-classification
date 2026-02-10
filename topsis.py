import numpy as np
import pandas as pd

# -------------------------------
# Load decision matrix
# -------------------------------
df = pd.read_csv("results/decision_matrix.csv")

models = df["Model"].values
data = df.drop(columns=["Model"]).values.astype(float)

# -------------------------------
# TOPSIS parameters
# -------------------------------
weights = np.array([0.3, 0.3, 0.2, 0.2])  # Accuracy, F1, Time, Size
criteria = ["benefit", "benefit", "cost", "cost"]

# -------------------------------
# Step 1: Normalize
# -------------------------------
norm_data = data / np.sqrt((data ** 2).sum(axis=0))

# -------------------------------
# Step 2: Weighted normalized matrix
# -------------------------------
weighted_data = norm_data * weights

# -------------------------------
# Step 3: Ideal best & worst
# -------------------------------
ideal_best = np.zeros(weighted_data.shape[1])
ideal_worst = np.zeros(weighted_data.shape[1])

for i in range(weighted_data.shape[1]):
    if criteria[i] == "benefit":
        ideal_best[i] = weighted_data[:, i].max()
        ideal_worst[i] = weighted_data[:, i].min()
    else:
        ideal_best[i] = weighted_data[:, i].min()
        ideal_worst[i] = weighted_data[:, i].max()

# -------------------------------
# Step 4: Distances
# -------------------------------
dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

# -------------------------------
# Step 5: TOPSIS score
# -------------------------------
scores = dist_worst / (dist_best + dist_worst)

# -------------------------------
# Ranking
# -------------------------------
rankings = scores.argsort()[::-1] + 1

result_df = pd.DataFrame({
    "Model": models,
    "TOPSIS_Score": scores,
    "Rank": rankings
}).sort_values("Rank")

result_df.to_csv("results/topsis_scores.csv", index=False)

print("\nTOPSIS Ranking:")
print(result_df)
