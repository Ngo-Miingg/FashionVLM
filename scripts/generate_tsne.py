import os
import torch
from transformers import AutoProcessor, AutoModel
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading dependencies and model...")
model_path = r"weights\v3_final\CLEAR_FashionVLM_V3_FINAL_MODEL"
csv_path = r"dataset_raw\data.csv"
img_dir = r"dataset_raw\data"
output_path = r"paper_latex\figures\tsne_plot.png"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and processor
try:
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Read dataset and sample categories
print("Reading dataset...")
df = pd.read_csv(csv_path)

# Filter specific categories to see clear clusters
target_categories = ['Sports Shoes', 'Shorts', 'Handbags', 'Watches']
df_filtered = df[df['category'].isin(target_categories)]

# Take 100 samples from each category (or max available) to avoid OOM/Timeout
samples = []
for cat in target_categories:
    cat_df = df_filtered[df_filtered['category'] == cat].head(75)
    samples.append(cat_df)

df_sample = pd.concat(samples).reset_index(drop=True)
print(f"Sampled {len(df_sample)} items for t-SNE.")

embeddings = []
labels = []

print("Extracting image features...")
with torch.no_grad():
    for idx, row in df_sample.iterrows():
        img_path = os.path.join(img_dir, str(row['image']))
        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            # Get image features from SigLIP
            out = model.get_image_features(**inputs)
            if hasattr(out, 'cpu'):
                image_features = out
            elif hasattr(out, 'pooler_output'):
                image_features = out.pooler_output
            elif isinstance(out, tuple):
                image_features = out[0]
            else:
                image_features = out
            embeddings.append(image_features.cpu().numpy().flatten())
            labels.append(row['category'])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if len(embeddings) == 0:
    print("No embeddings extracted. Exiting.")
    exit(1)

print("Running t-SNE...")
import numpy as np
embeddings_np = np.array(embeddings)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
tsne_results = tsne.fit_transform(embeddings_np)

print("Plotting results...")
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=tsne_results[:, 0], y=tsne_results[:, 1],
    hue=labels,
    palette=sns.color_palette("hsv", len(target_categories)),
    s=80, alpha=0.8, edgecolor='w', linewidth=0.5
)
plt.title("t-SNE Visualization of FashionVLM Latent Space", fontsize=16, fontweight='bold')
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.legend(title="Category", title_fontsize='13', fontsize='11', loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"t-SNE plot successfully saved to {output_path}")
