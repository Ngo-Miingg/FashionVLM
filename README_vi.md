*Đọc bằng ngôn ngữ khác: [English](README.md) | [Tiếng Việt](README_vi.md)*

# 👕 FashionVLM: Nâng cao Phân loại Thời trang Đa phương thức

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org/)

**Kho lưu trữ chính thức cho bài báo:** *"Enhancing Multimodal Fashion Classification Through Disentangled Learning Rates and Hard-Negative Contrastive Optimization"* 
**Tác giả:** Ngô Văn Minh (Đại học Đại Nam)

## 📌 Tổng quan
**FashionVLM** là một bộ khung đa phương thức được tối ưu hóa cao độ, xây dựng dựa trên kiến trúc **SigLIP2 Base**. Việc tinh chỉnh (fine-tuning) các Mô hình Ngôn ngữ - Thị giác (VLMs) khổng lồ trên các lĩnh vực hẹp như thương mại điện tử thời trang thường dẫn đến hiện tượng "quên thảm khốc" (catastrophic forgetting) và suy sụp biểu diễn (representation collapse). 

Để giải quyết vấn đề này, FashionVLM giới thiệu:
1. **Động lực học Tốc độ học Tách biệt (Disentangled Learning Rate Dynamics):** Tối ưu hóa bất đối xứng (Learning Rate của Text $\ll$ Learning Rate của Vision) giúp bảo tồn kiến thức ngôn ngữ nền tảng trong khi thích ứng mạnh mẽ với các kết cấu hình ảnh thời trang mới.
2. **Hàm Loss Contrastive Hard-Negative Phân theo Danh mục:** Một hình phạt toán học áp dụng cho các cặp âm bản nằm trong cùng một danh mục (VD: ép mô hình phải phân biệt được hai chiếc giày đen khác nhau) nhằm tăng cường độ nhạy bén của các đặc trưng hạt mịn (fine-grained features).

**Kết quả:** FashionVLM đạt **41.11% Within-Category Recall@1** (State-of-the-Art), vượt xa các mô hình lớn gấp 3 lần như OpenAI CLIP ViT-L/14 và OpenCLIP EVA02-L/14.

---

## 📂 Cấu trúc dự án
```text
FashionVLM_Project/
├── src/
│   └── train_siglip2.py       # Vòng lặp huấn luyện lõi với AdamW, Disentangled LR, & Hard-Neg Loss
├── scripts/
│   ├── generate_tsne.py       # Trích xuất embeddings và Trực quan hóa không gian tiềm ẩn bằng t-SNE
│   ├── generate_heatmap.py    # Tính ma trận tương đồng Cosine chéo giữa Ảnh và Text
│   └── generate_dist.py       # Phân tích phân phối chiều dài từ vựng của Dataset
├── paper_latex/               # Toàn bộ mã nguồn LaTeX của bài báo nghiên cứu + Hình ảnh
├── requirements.txt           # Danh sách thư viện Python
└── README.md                  # File này
```

---

## 📊 Kết quả cốt lõi (Tập dữ liệu Thời trang Kaggle)
| Kiến trúc | Chiến lược | Tham số | Standard R@1 | Within-Cat R@1 |
| :--- | :--- | :---: | :---: | :---: |
| CLIP ViT-L/14 | Tinh chỉnh (Fine-Tuned) | 427M | 33.03% | - |
| EVA02-L/14 | Tinh chỉnh (Fine-Tuned) | 427M | 36.11%* | - |
| **SigLIP2 Base** | Mặc định (Zero-Shot) | ~150M | 36.36% | 39.10% |
| **FashionVLM (Của chúng tôi)**| **Disentangled + Hard-Neg**| **~150M** | **38.97%** | **41.11%** |

*(EVA02-L/14 bị suy sụp biểu diễn trong quá trình tinh chỉnh, không thể vượt qua điểm số zero-shot ban đầu).*

---

## 🛠️ Cài đặt & Sử dụng

### 1. Thiết lập Môi trường
```bash
git clone https://github.com/Ngo-Miingg/FashionVLM.git
cd FashionVLM
pip install -r requirements.txt
```

### 2. Trực quan hóa Đặc trưng (t-SNE & Heatmaps)
Giả định rằng bạn đã tải bộ trọng số đã huấn luyện (pre-trained weights) vào thư mục `./weights` và tập dữ liệu vào `./dataset_raw`:
```bash
python scripts/generate_tsne.py
python scripts/generate_heatmap.py
```
*(Hãy kiểm tra thư mục `paper_latex/figures/` để xem các biểu đồ chất lượng cao được xuất ra).*

---

## 📖 Trích dẫn
Nếu bạn thấy mã nguồn hoặc bài báo này hữu ích cho nghiên cứu của mình, vui lòng trích dẫn:
```bibtex
@inproceedings{ngo2026fashionvlm,
  title={Enhancing Multimodal Fashion Classification Through Disentangled Learning Rates and Hard-Negative Contrastive Optimization},
  author={Ngo Van Minh},
  booktitle={Under Review},
  year={2026}
}
```