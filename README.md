<div align="center">
  <a href="#-english-version">🇺🇸 English</a> | <a href="#-phiên-bản-tiếng-việt">🇻🇳 Tiếng Việt</a>
</div>

## 🇺🇸 English Version
# 👕 FashionVLM: Enhancing Multimodal Fashion Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org/)

**Official Repository for the Paper:** *"Enhancing Multimodal Fashion Classification Through Disentangled Learning Rates and Hard-Negative Contrastive Optimization"* 
**Author:** Ngo Van Minh (Dai Nam University)

### 📌 Overview
**FashionVLM** is a highly optimized multimodal framework built upon the **SigLIP2 Base** architecture. Fine-tuning massive Vision-Language Models (VLMs) on narrow domains like e-commerce fashion often leads to catastrophic forgetting and representation collapse. 

To solve this, FashionVLM introduces:
1. **Disentangled Learning Rate Dynamics:** Asymmetric optimization (Text LR $\ll$ Vision LR) to preserve linguistic knowledge while adapting to new visual textures.
2. **Category-Aware Hard-Negative Contrastive Loss:** A mathematical penalty applied to intra-category negative pairs to force fine-grained visual discrimination.

**Result:** FashionVLM achieves State-of-the-Art **41.11% Within-Category Recall@1**, vastly outperforming 3x larger models like OpenAI CLIP ViT-L/14 and OpenCLIP EVA02-L/14.

### 📂 Project Structure
`	ext
FashionVLM_Project/
├── src/
│   └── train_siglip2.py       # Core training loop with AdamW, Disentangled LR, & Hard-Neg Loss
├── scripts/
│   ├── generate_tsne.py       # Extract embeddings and visualize Latent Space via t-SNE
│   ├── generate_heatmap.py    # Cross-modal Image-Text Cosine Similarity
│   └── generate_dist.py       # Dataset Token Length Distribution analysis
├── paper_latex/               # Full LaTeX source code of the research paper + Figures
├── requirements.txt           # Python dependencies
└── .gitignore                 
`

### 📊 Core Results (Kaggle Fashion Dataset)
| Architecture | Strategy | Params | Standard R@1 | Within-Cat R@1 |
| :--- | :--- | :---: | :---: | :---: |
| CLIP ViT-L/14 | Fine-Tuned | 427M | 33.03% | - |
| EVA02-L/14 | Fine-Tuned | 427M | 36.11%* | - |
| **SigLIP2 Base** | Zero-Shot | ~150M | 36.36% | 39.10% |
| **FashionVLM (Ours)**| **Disentangled + Hard-Neg**| **~150M** | **38.97%** | **41.11%** |

### 🛠️ Installation & Usage
`ash
git clone https://github.com/Ngo-Miingg/FashionVLM.git
cd FashionVLM
pip install -r requirements.txt
python scripts/generate_tsne.py
`

### 📖 Citation
`ibtex
@inproceedings{ngo2026fashionvlm,
  title={Enhancing Multimodal Fashion Classification Through Disentangled Learning Rates and Hard-Negative Contrastive Optimization},
  author={Ngo Van Minh},
  booktitle={Under Review},
  year={2026}
}
`

<br>
<hr>
<br>

## 🇻🇳 Phiên bản Tiếng Việt
# 👕 FashionVLM: Nâng cao Phân loại Thời trang Đa phương thức

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org/)

**Kho lưu trữ chính thức cho bài báo:** *"Enhancing Multimodal Fashion Classification Through Disentangled Learning Rates and Hard-Negative Contrastive Optimization"* 
**Tác giả:** Ngô Văn Minh (Đại học Đại Nam)

### 📌 Tổng quan
**FashionVLM** là một bộ khung đa phương thức được tối ưu hóa cao độ, xây dựng dựa trên kiến trúc **SigLIP2 Base**. Việc tinh chỉnh (fine-tuning) các VLMs khổng lồ trên các lĩnh vực hẹp như thời trang thường dẫn đến hiện tượng "quên thảm khốc" và suy sụp biểu diễn. 

Để giải quyết vấn đề này, FashionVLM giới thiệu:
1. **Động lực học Tốc độ học Tách biệt:** Tối ưu hóa bất đối xứng (LR Text $\ll$ LR Vision) giúp bảo tồn kiến thức ngôn ngữ trong khi thích ứng mạnh mẽ với kết cấu hình ảnh mới.
2. **Hàm Loss Contrastive Hard-Negative Theo Danh Mục:** Hình phạt toán học áp dụng cho các cặp âm bản trong cùng danh mục nhằm tăng cường độ nhạy bén của các đặc trưng hạt mịn.

**Kết quả:** FashionVLM đạt **41.11% Within-Category Recall@1** (SOTA), vượt xa các mô hình lớn gấp 3 lần như ViT-L/14 và EVA02-L/14.

### 📂 Cấu trúc dự án
`	ext
FashionVLM_Project/
├── src/
│   └── train_siglip2.py       # Vòng lặp huấn luyện lõi
├── scripts/
│   ├── generate_tsne.py       # Vẽ bản đồ t-SNE không gian tiềm ẩn
│   ├── generate_heatmap.py    # Tính ma trận tương đồng Cosine
│   └── generate_dist.py       # Phân bố độ dài từ vựng
├── paper_latex/               # Toàn bộ mã nguồn LaTeX
├── requirements.txt           
└── .gitignore                 
`

### 📊 Kết quả cốt lõi (Tập dữ liệu Kaggle)
| Kiến trúc | Chiến lược | Tham số | Standard R@1 | Within-Cat R@1 |
| :--- | :--- | :---: | :---: | :---: |
| CLIP ViT-L/14 | Tinh chỉnh | 427M | 33.03% | - |
| EVA02-L/14 | Tinh chỉnh | 427M | 36.11%* | - |
| **SigLIP2 Base** | Zero-Shot | ~150M | 36.36% | 39.10% |
| **FashionVLM (Ours)**| **Disentangled + Hard-Neg**| **~150M** | **38.97%** | **41.11%** |

### 🛠️ Cài đặt & Sử dụng
`ash
git clone https://github.com/Ngo-Miingg/FashionVLM.git
cd FashionVLM
pip install -r requirements.txt
python scripts/generate_tsne.py
`

### 📖 Trích dẫn
`ibtex
@inproceedings{ngo2026fashionvlm,
  title={Enhancing Multimodal Fashion Classification Through Disentangled Learning Rates and Hard-Negative Contrastive Optimization},
  author={Ngo Van Minh},
  booktitle={Under Review},
  year={2026}
}
`
<br>
<div align="center">
  <a href="#-english-version">⬆️ Back to Top</a>
</div>
