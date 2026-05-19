import os
import math
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    from visionscreen import VisionMultiscreenSegmentation
except ImportError:
    from vision_screen.visionscreen import VisionMultiscreenSegmentation

# ---------------------------------------------------------------------------
# Joint Transforms for Image and Mask (Native PyTorch/NumPy)
# ---------------------------------------------------------------------------
class JointTransform:
    def __init__(self, is_train=True):
        self.is_train = is_train
        
    def __call__(self, img, mask):
        # Convert PIL image (RGB) to numpy and then to Tensor
        img_np = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
        
        # Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Convert mask to torch tensor
        mask_tensor = torch.from_numpy(mask).long()
        
        if self.is_train:
            # Random horizontal flip (flip along width / last dimension)
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[-1])
                mask_tensor = torch.flip(mask_tensor, dims=[-1])
                
        return img_tensor, mask_tensor

# ---------------------------------------------------------------------------
# Cityscapes Dataset with BGR-to-Class mapping
# ---------------------------------------------------------------------------
class CityscapesDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, quick_run=False, quick_size=50):
        self.img_dir = os.path.join(data_dir, split, 'img')
        self.label_dir = os.path.join(data_dir, split, 'label')
        self.transform = transform
        
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Directory {self.img_dir} does not exist.")
            
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
        
        if quick_run:
            random.seed(42)
            self.img_names = random.sample(self.img_names, min(quick_size, len(self.img_names)))
            
        # Standard Cityscapes classes (RGB)
        colors_rgb = {
            'unlabeled': (0, 0, 0),
            'road': (128, 64, 128),
            'sidewalk': (244, 35, 232),
            'building': (70, 70, 70),
            'wall': (102, 102, 156),
            'fence': (190, 153, 153),
            'pole': (153, 153, 153),
            'traffic_light': (250, 170, 30),
            'traffic_sign': (220, 220, 0),
            'vegetation': (107, 142, 35),
            'terrain': (152, 251, 152),
            'sky': (70, 130, 180),
            'person': (220, 20, 60),
            'rider': (255, 0, 0),
            'car': (0, 0, 142),
            'truck': (0, 0, 70),
            'bus': (0, 60, 100),
            'train': (0, 80, 100),
            'motorcycle': (0, 0, 230),
            'bicycle': (119, 11, 32),
        }
        self.class_names = list(colors_rgb.keys())
        palette = np.array([colors_rgb[name] for name in self.class_names]) # (20, 3) in RGB
        
        # Swap R and B channels to get BGR palette because label images are stored in BGR representation
        self.palette_bgr = palette.copy()
        self.palette_bgr[:, [0, 2]] = self.palette_bgr[:, [2, 0]]
        
    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        label_img = Image.open(label_path)
        
        # Map label_img (BGR) to class indices (H, W) using Euclidean distance
        label_arr = np.array(label_img)
        flat_label = label_arr.reshape(-1, 3).astype(np.float32)
        dists = np.linalg.norm(flat_label[:, None, :] - self.palette_bgr[None, :, :], axis=2)
        mask = np.argmin(dists, axis=1).reshape(label_arr.shape[0], label_arr.shape[1])
        
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            img_np = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img_np).permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
            mask = torch.from_numpy(mask).long()
            
        return img, mask

# ---------------------------------------------------------------------------
# Mean Intersection-Over-Union (mIoU) Tracker
# ---------------------------------------------------------------------------
class IoUMetric:
    def __init__(self, num_classes=20, ignore_index=0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
        
    def reset(self):
        self.total_inter = np.zeros(self.num_classes)
        self.total_union = np.zeros(self.num_classes)
        
    def update(self, preds, targets):
        preds = preds.flatten()
        targets = targets.flatten()
        
        # Filter out ignored class
        keep = (targets != self.ignore_index)
        preds = preds[keep]
        targets = targets[keep]
        
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            pred_inds = (preds == cls)
            target_inds = (targets == cls)
            inter = np.logical_and(pred_inds, target_inds).sum()
            union = np.logical_or(pred_inds, target_inds).sum()
            self.total_inter[cls] += inter
            self.total_union[cls] += union
            
    def compute(self):
        ious = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            if self.total_union[cls] == 0:
                continue
            ious.append(self.total_inter[cls] / self.total_union[cls])
        return np.mean(ious) if len(ious) > 0 else 0.0

# ---------------------------------------------------------------------------
# Main Training regime
# ---------------------------------------------------------------------------
def train_model(args):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # Setup Datasets & DataLoaders
    print("Loading datasets...")
    train_transform = JointTransform(is_train=True)
    val_transform = JointTransform(is_train=False)
    
    train_dataset = CityscapesDataset(
        data_dir=args.data_dir, split='train', transform=train_transform,
        quick_run=args.quick_run, quick_size=args.quick_size
    )
    val_dataset = CityscapesDataset(
        data_dir=args.data_dir, split='val', transform=val_transform,
        quick_run=args.quick_run, quick_size=max(10, args.quick_size // 5)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train dataset size: {len(train_dataset)} images")
    print(f"Validation dataset size: {len(val_dataset)} images")
    
    # Model configuration
    # Small segmentation model based on VisionMultiscreen
    model = VisionMultiscreenSegmentation(
        img_size=(96, 256),
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=20,
        d_e=args.d_e,
        n_l=args.n_l,
        n_h=args.n_h,
        d_k=args.d_k,
        d_v=args.d_v,
        w_th=args.w_th
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine Annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_miou = 0.0
    iou_tracker = IoUMetric(num_classes=20, ignore_index=0)
    
    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images) # (B, 20, H, W)
            
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Batch Loss: {loss.item():.4f}")
                      
        scheduler.step()
        epoch_train_loss = train_loss / len(train_dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        iou_tracker.reset()
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()
                iou_tracker.update(preds, targets)
                
        epoch_val_loss = val_loss / len(val_dataset)
        val_miou = iou_tracker.compute()
        
        print(f"\n================ Epoch {epoch+1} Summary ================")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f}")
        print(f"Val mIoU:   {val_miou * 100:.2f}%")
        print("====================================================\n")
        
        # Save Checkpoint
        if val_miou > best_miou:
            best_miou = val_miou
            checkpoint_path = os.path.join(args.save_dir, "best_segmentation_model.pth")
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
            }, checkpoint_path)
            print(f"New best model saved with Val mIoU: {val_miou * 100:.2f}%\n")
            
    print(f"Training completed. Best Val mIoU: {best_miou * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Multiscreen Training on Cityscapes")
    parser.add_argument("--data-dir", type=str, default="cityscapes", help="Path to Cityscapes dataset folder")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer weight decay")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size for vision model")
    parser.add_argument("--w-th", type=float, default=256.0, help="Screening threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=20, help="Logging interval for batches")
    
    # Model size configurations (small scale default)
    parser.add_argument("--d-e", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-l", type=int, default=4, help="Number of layers")
    parser.add_argument("--n-h", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d-k", type=int, default=16, help="Key/Query dimension")
    parser.add_argument("--d-v", type=int, default=32, help="Value dimension")
    
    # Fast testing arguments
    parser.add_argument("--quick-run", action="store_true", help="Run a quick training validation on a small subset")
    parser.add_argument("--quick-size", type=int, default=50, help="Subset size for quick run")
    
    args = parser.parse_args()
    train_model(args)
