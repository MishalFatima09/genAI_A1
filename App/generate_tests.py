import numpy as np
from PIL import Image, ImageFilter
import os
import random
import glob

# Get a random image from the local Oxford-IIIT Pet dataset
dataset_path = r"D:\d driver\gen ai\oxford-iiit-pet\images"
all_images = glob.glob(os.path.join(dataset_path, "*.jpg"))
img_path = random.choice(all_images)
print(f"Selected: {os.path.basename(img_path)}")

# Resize to 128x128 exactly as the AI expects
img = Image.open(img_path).convert("RGB").resize((128, 128))
img.save("test_clean.png")

img_np = np.array(img)

# 1. Salt & Pepper Corruption
snp_np = img_np.copy()
prob = 0.1 # 10% of pixels corrupted
noise = np.random.rand(128, 128)
snp_np[noise < prob/2] = [0, 0, 0]
snp_np[(noise >= prob/2) & (noise < prob)] = [255, 255, 255]
Image.fromarray(snp_np).save("test_salt_pepper.png")

# 2. Occlusion Corruption (Draw 2 black rectangles)
occ_np = img_np.copy()
for _ in range(2):
    h, w = np.random.randint(20, 35, size=2)
    y, x = np.random.randint(0, 128 - h), np.random.randint(0, 128 - w)
    occ_np[y:y+h, x:x+w] = [0, 0, 0]
Image.fromarray(occ_np).save("test_occlusion.png")

# 3. Gaussian Blur Corruption
img_blur = img.filter(ImageFilter.GaussianBlur(radius=1.5))
img_blur.save("test_blur.png")

print("Successfully created mathematical test images from Oxford Dataset!")
