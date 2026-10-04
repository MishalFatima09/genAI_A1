import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image

def apply_salt_and_pepper(img_tensor):
    p = random.uniform(0.02, 0.15)
    mask = torch.rand(img_tensor.shape[1:]) # [H, W]
    corrupted = img_tensor.clone()
    corrupted[:, mask < p/2] = 1.0 # Salt (White)
    corrupted[:, (mask >= p/2) & (mask < p)] = 0.0 # Pepper (Black)
    return corrupted

def apply_gaussian_blur(img_tensor):
    k = random.choice([3, 5, 7])
    sigma = random.uniform(0.5, 2.5)
    return TF.gaussian_blur(img_tensor, kernel_size=[k, k], sigma=[sigma, sigma])

def apply_occlusion(img_tensor):
    c, h, w = img_tensor.shape
    num_rects = random.randint(1, 3)
    target_area_ratio = random.uniform(0.10, 0.35)
    total_area = h * w
    area_per_rect = (target_area_ratio * total_area) / num_rects
    
    corrupted = img_tensor.clone()
    for _ in range(num_rects):
        aspect_ratio = random.uniform(0.5, 2.0)
        rect_h = max(1, min(h, int((area_per_rect * aspect_ratio) ** 0.5)))
        rect_w = max(1, min(w, int((area_per_rect / aspect_ratio) ** 0.5)))
        
        top = random.randint(0, max(1, h - rect_h))
        left = random.randint(0, max(1, w - rect_w))
        
        corrupted[:, top:top+rect_h, left:left+rect_w] = 0.0 # Black out the region
    return corrupted

def apply_random_corruption(img_tensor, is_training=True, seed=None):
    """
    Applies one of the four corruptions dynamically.
    For validation, we pass a seed to make the corruption deterministic (manifest).
    """
    if not is_training and seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        
    choice = random.randint(0, 3)
    
    if choice == 0:
        corrupted = img_tensor.clone()
        label = "Clean"
    elif choice == 1:
        corrupted = apply_salt_and_pepper(img_tensor)
        label = "Salt & Pepper"
    elif choice == 2:
        corrupted = apply_gaussian_blur(img_tensor)
        label = "Gaussian Blur"
    else:
        corrupted = apply_occlusion(img_tensor)
        label = "Occlusion"
        
    # Reset seed to random if we fixed it for validation
    if not is_training:
        random.seed()
        torch.seed()
        
    return corrupted, choice, label

class PetCorruptionDataset(Dataset):
    def __init__(self, root_dir, is_training=True):
        # Base transform: Resize to 128x128 and convert to Tensor [0, 1]
        self.base_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # We look for all .jpg files in the images folder
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.jpg")))
        if len(self.image_paths) == 0:
             print(f"Warning: No images found in {os.path.join(root_dir, 'images')}")
        
        # The assignment requires an 80/20 train/val split using seed 42.
        dataset_size = len(self.image_paths)
        indices = list(range(dataset_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(np.floor(0.2 * dataset_size)) # 20% for val
        
        if is_training:
            self.indices = indices[split_idx:]
        else:
            self.indices = indices[:split_idx]
            
        self.is_training = is_training
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path = self.image_paths[actual_idx]
        
        # Load image and convert to RGB (some might be grayscale or have alpha channel)
        clean_img = Image.open(img_path).convert("RGB")
        clean_tensor = self.base_transform(clean_img)
        
        # For validation, we use the actual dataset index as a seed so it is completely deterministic
        seed = None if self.is_training else actual_idx
        
        corrupted_tensor, corruption_idx, corruption_label = apply_random_corruption(clean_tensor, self.is_training, seed)
        
        return corrupted_tensor, clean_tensor, corruption_label

if __name__ == "__main__":
    print("Testing the dataset pipeline...")
    # Because you have the dataset locally, we can point it to the local folder for testing
    local_data_dir = "./oxford-iiit-pet"
    
    if os.path.exists(local_data_dir):
        dataset = PetCorruptionDataset(root_dir=local_data_dir, is_training=True)
        print(f"Loaded {len(dataset)} training images.")
        
        if len(dataset) > 0:
            corrupted, clean, label = dataset[0]
            print(f"Sample 0: Label={label}, Corrupted Shape={corrupted.shape}, Clean Shape={clean.shape}")
    else:
        print(f"Could not find dataset at {local_data_dir}. Ensure 'images' folder is inside it.")
