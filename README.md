<div align="right">
  <strong>🇺🇸 English</strong> | <a href="README_vi.md">🇻🇳 Tiếng Việt</a>
</div>

# 👕 FashionVLM: Enhancing Multimodal Fashion Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org/)

**Official Repository for the Paper:** *"Enhancing Multimodal Fashion Classification Through Disentangled Learning Rates and Hard-Negative Contrastive Optimization"* 
**Author:** Ngo Van Minh (Dai Nam University)

## 📌 Overview
**FashionVLM** is a highly optimized multimodal framework built upon the **SigLIP2 Base** architecture. Fine-tuning massive Vision-Language Models (VLMs) on narrow domains like e-commerce fashion often leads to catastrophic forgetting and representation collapse. 

To solve this, FashionVLM introduces:
1. **Disentangled Learning Rate Dynamics:** Asymmetric optimization (Text LR $\ll$ Vision LR) to preserve linguistic knowledge while adapting to new visual textures.
2. **Category-Aware Hard-Negative Contrastive Loss:** A mathematical penalty applied to intra-category negative pairs to force fine-grained visual discrimination.

**Result:** FashionVLM achieves State-of-the-Art **41.11% Within-Category Recall@1**, vastly outperforming 3x larger models like OpenAI CLIP ViT-L/14 and OpenCLIP EVA02-L/14.

---

## 📂 Project Structure
```text
FashionVLM_Project/
├── src/
│   └── train_siglip2.py       # Core training loop with AdamW, Disentangled LR, & Hard-Neg Loss
├── scripts/
│   ├── generate_tsne.py       # Extract embeddings and visualize Latent Space via t-SNE
│   ├── generate_heatmap.py    # Cross-modal Image-Text Cosine Similarity
│   └── generate_dist.py       # Dataset Token Length Distribution analysis
├── paper_latex/               # Full LaTeX source code of the research paper + Figures
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 📊 Core Results (Kaggle Fashion Dataset)
| Architecture | Strategy | Params | Standard R@1 | Within-Cat R@1 |
| :--- | :--- | :---: | :---: | :---: |
| CLIP ViT-L/14 | Fine-Tuned | 427M | 33.03% | - |
| EVA02-L/14 | Fine-Tuned | 427M | 36.11%* | - |
| **SigLIP2 Base** | Zero-Shot | ~150M | 36.36% | 39.10% |
| **FashionVLM (Ours)**| **Disentangled + Hard-Neg**| **~150M** | **38.97%** | **41.11%** |

*(EVA02-L/14 suffered from representation collapse during fine-tuning, plateauing at zero-shot metrics).*

---

## 🛠️ Installation & Usage

### 1. Setup Environment
```bash
git clone https://github.com/Ngo-Miingg/FashionVLM.git
cd FashionVLM
pip install -r requirements.txt
```

### 2. Feature Visualization (t-SNE & Heatmaps)
```bash
python scripts/generate_tsne.py
python scripts/generate_heatmap.py
```
*(Check the `paper_latex/figures/` folder for the generated high-quality plots).*

---

## 📖 Citation
If you find this code or our paper useful for your research, please consider citing:
```bibtex
@inproceedings{ngo2026fashionvlm,
  title={Enhancing Multimodal Fashion Classification Through Disentangled Learning Rates and Hard-Negative Contrastive Optimization},
  author={Ngo Van Minh},
  booktitle={Under Review},
  year={2026}
}
```