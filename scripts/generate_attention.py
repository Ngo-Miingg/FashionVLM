import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import AutoProcessor, AutoModel

# Đường dẫn
model_path = r"D:\Work\ABC\FashionVLM\weights\v3_final\CLEAR_FashionVLM_V3_FINAL_MODEL"
img_path = r"D:\Work\ABC\FashionVLM\dataset_raw\data\3238.jpg" # Giày Puma
out_path = r"D:\Work\ABC\FashionVLM_Project\paper_latex\figures\attention_map.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading model for Attention Map...")

try:
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
except Exception as e:
    print(f"Error: {e}")
    exit(1)

print("Processing image...")
image = Image.open(img_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    # Lấy output của Vision Encoder
    vision_outputs = model.vision_model(**inputs)
    # Lấy last_hidden_state (Shape: [1, seq_len, hidden_size] vd: [1, 256, 768])
    last_hidden_state = vision_outputs.last_hidden_state
    
    # Tính độ lớn kích hoạt (Activation Magnitude) của từng patch bằng L2 Norm
    # Bỏ qua CLS token nếu có (SigLIP thường ko dùng CLS, nhưng cứ an toàn kiểm tra)
    num_patches = last_hidden_state.shape[1]
    grid_size = int(np.sqrt(num_patches))
    
    if grid_size * grid_size != num_patches:
        # Nếu có CLS token (vd 257 patches -> bỏ patch đầu tiên)
        patch_states = last_hidden_state[0, 1:, :]
        grid_size = int(np.sqrt(num_patches - 1))
    else:
        patch_states = last_hidden_state[0, :, :]
        
    # Tính L2 norm theo chiều hidden (dim=1)
    activation_map = torch.norm(patch_states, dim=-1).cpu().numpy()
    
    # Reshape thành lưới 2D (ví dụ 16x16)
    activation_map = activation_map.reshape((grid_size, grid_size))

# Chuẩn hóa map về 0-255
activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
activation_map = np.uint8(255 * activation_map)

# Phóng to map lên bằng kích thước ảnh gốc (256x256)
img_cv = cv2.cvtColor(np.array(image.resize((256, 256))), cv2.COLOR_RGB2BGR)
heatmap = cv2.applyColorMap(cv2.resize(activation_map, (256, 256)), cv2.COLORMAP_JET)

# Trộn ảnh gốc và heatmap
superimposed_img = heatmap * 0.4 + img_cv * 0.6
cv2.imwrite(out_path, superimposed_img)

print(f"Attention Map saved to {out_path}")
