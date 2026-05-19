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
from torch.utils.data import DataLoader

try:
    from visionscreen import PatchEmbed
    from train_cityscapes import JointTransform, CityscapesDataset, IoUMetric
except ImportError:
    from vision_screen.visionscreen import PatchEmbed
    from vision_screen.train_cityscapes import JointTransform, CityscapesDataset, IoUMetric

# ---------------------------------------------------------------------------
# Standard Transformer Block for ViT
# ---------------------------------------------------------------------------
class ViTBlock(nn.Module):
    def __init__(self, d_e: int, n_h: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_e)
        self.attn = nn.MultiheadAttention(embed_dim=d_e, num_heads=n_h, batch_first=True)
        self.norm2 = nn.LayerNorm(d_e)
        self.mlp = nn.Sequential(
            nn.Linear(d_e, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_e)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------------------------------------------------------------------
# ViT Segmentation Model
# ---------------------------------------------------------------------------
class ViTSegmentation(nn.Module):
    """
    Standard Vision Transformer model adapted for Semantic Segmentation.
    """
    def __init__(self, img_size=(96, 256), patch_size=16, 
                 in_chans=3, num_classes=20,
                 d_e=128, n_l=4, n_h=4, d_ff=None):
        super().__init__()
        self.d_e = d_e
        self.n_l = n_l
        self.n_h = n_h
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, d_e=d_e
        )
        self.num_patches = self.patch_embed.num_patches
        
        # Standard ViT learned 1D positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_e))
        
        if d_ff is None:
            d_ff = 4 * d_e
            
        self.blocks = nn.ModuleList([
            ViTBlock(d_e, n_h, d_ff) for _ in range(n_l)
        ])
        
        self.norm = nn.LayerNorm(d_e)
        
        # Segmentation decoder matching Multiscreen exactly
        self.decoder_pred = nn.Sequential(
            nn.Conv2d(d_e, d_e, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_e),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_e, num_classes, kernel_size=1)
        )
        
        nn.init.normal_(self.pos_embed, std=0.02)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.patch_embed.proj.weight, std=0.02)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
            
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        Hp, Wp = self.patch_embed.grid_size
        
        # 1. Project patches and add positional embeddings
        x = self.patch_embed(img) # (B, N, d_e)
        x = x + self.pos_embed
        
        # 2. Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x) # (B, N, d_e)
        
        # 3. Reshape back to spatial feature grid: (B, N, d_e) -> (B, d_e, Hp, Wp)
        x = x.transpose(1, 2).view(B, self.d_e, Hp, Wp)
        
        # 4. Decoder prediction and bilinear upsample to input image size (H, W)
        feats = self.decoder_pred(x)
        logits = F.interpolate(feats, size=(H, W), mode='bilinear', align_corners=True)
        
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# Training Loop
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
    model = ViTSegmentation(
        img_size=(96, 256),
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=20,
        d_e=args.d_e,
        n_l=args.n_l,
        n_h=args.n_h,
        d_ff=args.d_ff
    ).to(device)
    
    print(f"ViT model parameters: {model.count_parameters():,}")
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine Annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_miou = 0.0
    iou_tracker = IoUMetric(num_classes=20, ignore_index=0)
    
    print("\n--- Starting ViT Training ---")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            
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
            checkpoint_path = os.path.join(args.save_dir, "best_vit_model.pth")
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
    parser = argparse.ArgumentParser(description="ViT Training on Cityscapes")
    parser.add_argument("--data-dir", type=str, default="cityscapes", help="Path to Cityscapes dataset folder")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer weight decay")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size for vision model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=20, help="Logging interval for batches")
    
    # Model size configurations (small scale default matching Multiscreen)
    parser.add_argument("--d-e", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-l", type=int, default=4, help="Number of layers")
    parser.add_argument("--n-h", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=512, help="Feedforward dimension (defaults to 4 * d_e)")
    
    # Fast testing arguments
    parser.add_argument("--quick-run", action="store_true", help="Run a quick training validation on a small subset")
    parser.add_argument("--quick-size", type=int, default=50, help="Subset size for quick run")
    
    args = parser.parse_args()
    train_model(args)
