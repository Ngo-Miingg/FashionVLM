import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Reading dataset...")
csv_path = r"dataset_raw\data.csv"
df = pd.read_csv(csv_path)

# Calculate word counts
df['title_len'] = df['display name'].apply(lambda x: len(str(x).split()))
df['desc_len'] = df['description'].apply(lambda x: len(str(x).split()))

print("Plotting distribution...")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df['title_len'], fill=True, label="Short Query (Display Name)", color="blue", alpha=0.5)
sns.kdeplot(data=df['desc_len'], fill=True, label="Long Query (Description)", color="orange", alpha=0.5)

plt.axvline(x=64, color='red', linestyle='--', linewidth=2, label="Max Token Length Cutoff (64)")

plt.title("Text Length Distribution in Fashion Dataset", fontsize=15, fontweight='bold')
plt.xlabel("Number of Words", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(0, 200)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

out_path = r"paper_latex\figures\text_length_dist.png"
plt.savefig(out_path, dpi=300)
print(f"Distribution plot saved to {out_path}")
