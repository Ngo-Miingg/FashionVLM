import os
import torch
from transformers import AutoProcessor, AutoModel
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

print("Loading dependencies and model...")
model_path = r"weights\v3_final\CLEAR_FashionVLM_V3_FINAL_MODEL"
csv_path = r"dataset_raw\data.csv"
img_dir = r"dataset_raw\data"
out_path = r"paper_latex\figures\similarity_heatmap.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print("Reading dataset and selecting diverse samples...")
df = pd.read_csv(csv_path).dropna(subset=['image', 'display name'])

# Select 5 diverse categories
cats = ['Sports Shoes', 'Handbags', 'Watches', 'Tshirts', 'Sunglasses']
samples = []
for c in cats:
    match = df[df['category'] == c]
    if len(match) > 0:
        samples.append(match.iloc[0])

if len(samples) < 5:
    samples = df.sample(5).to_dict('records')
    
images = []
texts = []
labels = []

for s in samples:
    img_path = os.path.join(img_dir, str(s['image']))
    if os.path.exists(img_path):
        images.append(Image.open(img_path).convert("RGB"))
        texts.append(str(s['display name']))
        labels.append(str(s['category']))

if len(images) == 0:
    print("No images found.")
    exit(1)

print("Computing features...")
with torch.no_grad():
    inputs = processor(text=texts, images=images, padding="max_length", return_tensors="pt").to(device)
    out = model(**inputs)
    
    # Get embeddings and normalize
    img_embeds = out.image_embeds
    txt_embeds = out.text_embeds
    
    img_embeds = F.normalize(img_embeds, p=2, dim=-1)
    txt_embeds = F.normalize(txt_embeds, p=2, dim=-1)
    
    # Compute dot product (cosine similarity since normalized)
    similarity = torch.matmul(img_embeds, txt_embeds.t()).cpu().numpy()

print("Plotting heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(
    similarity, annot=True, fmt=".2f", cmap="YlGnBu",
    xticklabels=[t[:15]+"..." for t in texts],
    yticklabels=[f"Img: {l}" for l in labels]
)
plt.title("Cross-Modal Cosine Similarity (Image-to-Text)", fontsize=14, fontweight='bold')
plt.xlabel("Text Queries", fontsize=12)
plt.ylabel("Images", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig(out_path, dpi=300)
print(f"Heatmap saved to {out_path}")
