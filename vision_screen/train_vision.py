import torch
import torch.nn as nn
import torch.optim as optim
from visionscreen import VisionMultiscreen

def main():
    # Set device to CPU to avoid ROCm/HIP driver mismatch errors
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model configuration
    # Small scale model for testing
    img_size = 224
    patch_size = 16
    in_chans = 3
    num_classes = 10
    
    d_e = 128
    n_l = 2
    n_h = 4
    d_k = 16
    d_v = 32

    print("Initializing Vision Multiscreen model...")
    model = VisionMultiscreen(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        d_e=d_e,
        n_l=n_l,
        n_h=n_h,
        d_k=d_k,
        d_v=d_v
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Dummy inputs: Batch of 4 images
    dummy_images = torch.randn(4, in_chans, img_size, img_size, device=device)
    dummy_labels = torch.randint(0, num_classes, (4,), device=device)

    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    print("\n--- Running Forward Pass ---")
    logits = model(dummy_images)
    print(f"Input shape:  {dummy_images.shape}")
    print(f"Logits shape: {logits.shape}")
    
    loss = criterion(logits, dummy_labels)
    print(f"Loss value:   {loss.item():.4f}")

    print("\n--- Running Backward Pass & Optimization Step ---")
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients for some parameters
    has_grads = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Warning: {name} has no gradient!")
            has_grads = False
            
    if has_grads:
        print("All parameters successfully received gradients.")
        
    optimizer.step()
    print("Optimization step completed successfully.")

if __name__ == "__main__":
    main()
