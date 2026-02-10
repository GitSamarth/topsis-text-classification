import pandas as pd
import matplotlib.pyplot as plt
import os

# Create plots directory if not exists
os.makedirs("plots", exist_ok=True)

# Load TOPSIS results
df = pd.read_csv("results/topsis_scores.csv")

# Plot
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["TOPSIS_Score"])
plt.xlabel("Model")
plt.ylabel("TOPSIS Score")
plt.title("TOPSIS Ranking of Pretrained Text Classification Models")

# Save plot
plt.tight_layout()
plt.savefig("plots/topsis_bar_chart.png")
plt.show()
