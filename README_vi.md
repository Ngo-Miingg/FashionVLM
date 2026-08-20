<div align="right">
  <a href="README.md">🇺🇸 English</a> | <strong>🇻🇳 Tiếng Việt</strong>
</div>

# 👕 FashionVLM: Nâng cao Phân loại Thời trang Đa phương thức

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org/)

**Kho lưu trữ chính thức cho bài báo:** *"Enhancing Multimodal Fashion Classification Through Disentangled Learning Rates and Hard-Negative Contrastive Optimization"* 
**Tác giả:** Ngô Văn Minh (Đại học Đại Nam)

## 📌 Tổng quan
**FashionVLM** là một bộ khung đa phương thức được tối ưu hóa cao độ, xây dựng dựa trên kiến trúc **SigLIP2 Base**. Việc tinh chỉnh (fine-tuning) các VLMs khổng lồ trên các lĩnh vực hẹp như thương mại điện tử thường dẫn đến hiện tượng "quên thảm khốc" (catastrophic forgetting) và suy sụp biểu diễn (representation collapse). 

Để giải quyết vấn đề này, FashionVLM giới thiệu:
1. **Động lực học Tốc độ học Tách biệt (Disentangled LR):** Tối ưu hóa bất đối xứng (LR Text $\ll$ LR Vision) giúp bảo tồn kiến thức ngôn ngữ nền tảng trong khi thích ứng mạnh mẽ với kết cấu hình ảnh mới.
2. **Hàm Loss Contrastive Hard-Negative Theo Danh Mục:** Một hình phạt toán học áp dụng cho các cặp âm bản trong cùng danh mục nhằm tăng cường độ nhạy bén của các đặc trưng hạt mịn.

**Kết quả:** FashionVLM đạt **41.11% Within-Category Recall@1** (State-of-the-Art), vượt xa các mô hình lớn gấp 3 lần như ViT-L/14 và EVA02-L/14.

---

## 📂 Cấu trúc dự án
```text
FashionVLM_Project/
├── src/
│   └── train_siglip2.py       # Vòng lặp huấn luyện lõi
├── scripts/
│   ├── generate_tsne.py       # Vẽ bản đồ t-SNE không gian tiềm ẩn
│   ├── generate_heatmap.py    # Tính ma trận tương đồng Cosine
│   └── generate_dist.py       # Phân bố độ dài từ vựng
├── paper_latex/               # Toàn bộ mã nguồn LaTeX
├── requirements.txt           
└── README.md                  
```

---

## 📊 Kết quả cốt lõi (Tập dữ liệu Kaggle)
| Kiến trúc | Chiến lược | Tham số | Standard R@1 | Within-Cat R@1 |
| :--- | :--- | :---: | :---: | :---: |
| CLIP ViT-L/14 | Tinh chỉnh | 427M | 33.03% | - |
| EVA02-L/14 | Tinh chỉnh | 427M | 36.11%* | - |
| **SigLIP2 Base** | Zero-Shot | ~150M | 36.36% | 39.10% |
| **FashionVLM (Ours)**| **Disentangled + Hard-Neg**| **~150M** | **38.97%** | **41.11%** |

---

## 🛠️ Cài đặt & Sử dụng

### 1. Thiết lập Môi trường
```bash
git clone https://github.com/Ngo-Miingg/FashionVLM.git
cd FashionVLM
pip install -r requirements.txt
```

### 2. Trực quan hóa Đặc trưng
```bash
python scripts/generate_tsne.py
python scripts/generate_heatmap.py
```

---

## 📖 Trích dẫn
```bibtex
@inproceedings{ngo2026fashionvlm,
  title={Enhancing Multimodal Fashion Classification Through Disentangled Learning Rates and Hard-Negative Contrastive Optimization},
  author={Ngo Van Minh},
  booktitle={Under Review},
  year={2026}
}
```