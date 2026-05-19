import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers: Row-wise unit normalisation (RSS) and TanhNorm
# ---------------------------------------------------------------------------
def rss(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise unit-length normalisation (divide each row by its L2 norm)."""
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)

def tanh_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """TanhNorm(x) = tanh(||x||) / ||x|| * x. For bounding output norm."""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return (torch.tanh(norm) / norm) * x

# ---------------------------------------------------------------------------
# 2D MiPE - Minimal Positional Encoding adapted for Images
# ---------------------------------------------------------------------------
def apply_2d_mipe(x: torch.Tensor, x_pos: torch.Tensor, y_pos: torch.Tensor, 
                  w: torch.Tensor, w_th: float = 256.0) -> torch.Tensor:
    """
    Apply 2D MiPE rotation to the first four dimensions of x.
    - Channels 0,1 are rotated based on horizontal (x) patch coordinates.
    - Channels 2,3 are rotated based on vertical (y) patch coordinates.
    
    x: (..., N, d) or (B, H, N, d)
    x_pos: (N,)
    y_pos: (N,)
    w: (H,) or broadcastable shape
    """
    assert x.shape[-1] >= 4, "Key/Query dimension d_k must be >= 4 for 2D MiPE rotation."
    
    # Calculate decay factor gamma based on head screening windows w
    gamma = torch.where(
        w < w_th,
        (torch.cos(math.pi * w / w_th) + 1.0) / 2.0,
        torch.zeros_like(w),
    )
    gamma = gamma.unsqueeze(1)               # (H, 1)
    w_uns = w.unsqueeze(1).clamp(min=1e-8)   # (H, 1)

    # --- Horizontal Rotation (Channels 0 and 1) ---
    pos_x = x_pos.float().unsqueeze(0)        # (1, N)
    angle_x = (math.pi * pos_x * gamma) / w_uns  # (H, N)
    while angle_x.dim() < x.dim() - 1:
        angle_x = angle_x.unsqueeze(0)
    angle_x = angle_x.unsqueeze(-1)          # (1, H, N, 1)
    cos_ax, sin_ax = torch.cos(angle_x), torch.sin(angle_x)

    # --- Vertical Rotation (Channels 2 and 3) ---
    pos_y = y_pos.float().unsqueeze(0)        # (1, N)
    angle_y = (math.pi * pos_y * gamma) / w_uns  # (H, N)
    while angle_y.dim() < x.dim() - 1:
        angle_y = angle_y.unsqueeze(0)
    angle_y = angle_y.unsqueeze(-1)          # (1, H, N, 1)
    cos_ay, sin_ay = torch.cos(angle_y), torch.sin(angle_y)

    # Rotate channels in pairs
    x0, x1 = x[..., 0:1], x[..., 1:2]
    x2, x3 = x[..., 2:3], x[..., 3:4]
    x_rest = x[..., 4:]

    r0 = x0 * cos_ax - x1 * sin_ax
    r1 = x0 * sin_ax + x1 * cos_ax

    r2 = x2 * cos_ay - x3 * sin_ay
    r3 = x2 * sin_ay + x3 * cos_ay

    return torch.cat([r0, r1, r2, r3, x_rest], dim=-1)

# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Split 2D images into patches and project them to standard latent dimension."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_chans: int = 3, d_e: int = 768):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, d_e, kernel_size=self.patch_size, stride=self.patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}x{W}) does not match model expected size ({self.img_size[0]}x{self.img_size[1]})."
        x = self.proj(x)                  # (B, d_e, H_p, W_p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, d_e)
        return x

# ---------------------------------------------------------------------------
# Vision Multiscreen Layer (Bidirectional 2D Gated Screening)
# ---------------------------------------------------------------------------
class VisionMultiscreenLayer(nn.Module):
    def __init__(self, n_h: int, d_e: int, d_k: int, d_v: int,
                 w_th: float = 256.0, sw_values: list = None,
                 init_so: float = 0.0):
        super().__init__()
        self.n_h = n_h
        self.d_e = d_e
        self.d_k = d_k
        self.d_v = d_v
        self.w_th = w_th

        # Projections
        self.W_Q = nn.Linear(d_e, n_h * d_k, bias=False)
        self.W_K = nn.Linear(d_e, n_h * d_k, bias=False)
        self.W_V = nn.Linear(d_e, n_h * d_v, bias=False)
        self.W_G = nn.Linear(d_e, n_h * d_v, bias=False)
        self.W_O = nn.Linear(n_h * d_v, d_e, bias=False)

        # Learned scalars
        if sw_values is None:
            sw_values = [0.0] * n_h
        self.s_w = nn.Parameter(torch.tensor(sw_values))
        self.s_r = nn.Parameter(torch.zeros(n_h))
        self.s_O = nn.Parameter(torch.full((n_h,), init_so))

    def forward(self, x: torch.Tensor, x_pos: torch.Tensor, y_pos: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H = self.n_h

        # Projections: (B, N, d_e) -> (B, H, N, d_kv)
        q = self.W_Q(x).view(B, N, H, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(B, N, H, self.d_k).transpose(1, 2)
        v = self.W_V(x).view(B, N, H, self.d_v).transpose(1, 2)
        g = self.W_G(x).view(B, N, H, self.d_v).transpose(1, 2)

        # Compute screening window & acceptance width
        w = torch.exp(self.s_w) + 1.0
        r = torch.sigmoid(self.s_r)

        # Row-wise unit normalization
        q_bar = rss(q)
        k_bar = rss(k)
        v_bar = rss(v)

        # Apply 2D spatial MiPE rotation
        q_tilde = apply_2d_mipe(q_bar, x_pos, y_pos, w, self.w_th)
        k_tilde = apply_2d_mipe(k_bar, x_pos, y_pos, w, self.w_th)

        # Bounded cosine similarity: s_ij = q̃_i · k̃_j^T ∈ [-1, 1]
        sim = q_tilde @ k_tilde.transpose(-2, -1)  # (B, H, N, N)

        # Content Trim calculation
        r_uns = r.view(1, H, 1, 1).clamp(min=1e-8)
        alpha = torch.clamp(1.0 - (1.0 - sim) / r_uns, min=0.0)
        alpha = alpha ** 2

        # --- 2D Bidirectional Softmask (Symmetric Neighborhood decay) ---
        dx = x_pos.unsqueeze(0) - x_pos.unsqueeze(1)   # (N, N)
        dy = y_pos.unsqueeze(0) - y_pos.unsqueeze(1)   # (N, N)
        dist = torch.sqrt(dx**2 + dy**2)               # (N, N)
        
        w_uns = w.view(H, 1, 1)
        dist_uns = dist.unsqueeze(0)                   # (1, N, N)

        # Include everything within distance w
        in_window = dist_uns <= w_uns
        cos_mask = (torch.cos(math.pi * dist_uns / w_uns.clamp(min=1e-8)) + 1.0) / 2.0
        softmask = torch.where(in_window, cos_mask, torch.zeros_like(cos_mask))
        softmask = softmask.unsqueeze(0)               # (1, H, N, N)

        # Combine Trim (content) and Softmask (distance) relevance
        alpha_d = alpha * softmask

        # Aggregate values and apply TanhNorm
        h = alpha_d @ v_bar
        u = tanh_norm(h)

        # SiLU-tanh gating
        g_hat = torch.tanh(F.silu(g))

        # Scale layer output
        out = u * g_hat
        out = out * torch.exp(self.s_O).view(1, H, 1, 1)

        # Project back to original embedding dimension d_e
        out = out.transpose(1, 2).reshape(B, N, H * self.d_v)
        delta_x = self.W_O(out)
        
        return x + delta_x

# ---------------------------------------------------------------------------
# Vision Multiscreen Model
# ---------------------------------------------------------------------------
class VisionMultiscreen(nn.Module):
    """
    Vision Multiscreen model adapted for Image Classification.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_chans: int = 3, num_classes: int = 10,
                 d_e: int = 192, n_l: int = 12, n_h: int = 12, 
                 d_k: int = 16, d_v: int = 64, w_th: float = 256.0):
        super().__init__()
        self.d_e = d_e
        self.n_l = n_l
        self.n_h = n_h
        self.d_k = d_k
        self.d_v = d_v
        self.w_th = w_th
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, d_e=d_e
        )
        self.num_patches = self.patch_embed.num_patches
        
        # Scaling parameters
        self.s_E = nn.Parameter(torch.tensor(0.0))
        self.s_F = nn.Parameter(torch.tensor(math.log(math.sqrt(d_e))))
        
        # Paper-faithful unit-normalised output classifier weights
        self.classifier_weight = nn.Parameter(torch.empty(num_classes, d_e))
        
        # Linearly spaced s_w and initial layer scaling s_O
        sw_values = torch.linspace(0.0, math.log(w_th), n_h).tolist()
        init_so = math.log(1.0 / math.sqrt(n_h * n_l))
        
        self.layers = nn.ModuleList([
            VisionMultiscreenLayer(n_h, d_e, d_k, d_v, w_th=w_th,
                                   sw_values=sw_values, init_so=init_so)
            for _ in range(n_l)
        ])
        
        self._init_weights()

    def _init_weights(self):
        d_k, d_v, d_e = self.d_k, self.d_v, self.d_e
        
        # Initialise projections
        nn.init.normal_(self.patch_embed.proj.weight, std=0.02)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
            
        # Initialise classifier: N(0, 0.1/√d_E)
        nn.init.normal_(self.classifier_weight, std=0.1 / math.sqrt(d_e))
        
        for layer in self.layers:
            nn.init.normal_(layer.W_Q.weight, std=0.1 / math.sqrt(d_k))
            nn.init.normal_(layer.W_K.weight, std=0.1 / math.sqrt(d_k))
            nn.init.normal_(layer.W_V.weight, std=0.1 / math.sqrt(d_v))
            nn.init.normal_(layer.W_G.weight, std=0.1)
            nn.init.normal_(layer.W_O.weight, std=0.1 / math.sqrt(d_e))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, C, H, W)
        """
        B, C, H, W = img.shape
        
        # Generate 2D patch grid coordinates dynamically to match input device/size
        Hp, Wp = self.patch_embed.grid_size
        grid_y, grid_x = torch.meshgrid(
            torch.arange(Hp, device=img.device),
            torch.arange(Wp, device=img.device),
            indexing='ij'
        )
        x_pos = grid_x.reshape(-1) # (N,)
        y_pos = grid_y.reshape(-1) # (N,)
        
        # 1. Project patches and apply RSS
        emb = self.patch_embed(img) # (B, N, d_e)
        x = torch.exp(self.s_E) * rss(emb)
        
        # 2. Residual blocks
        for layer in self.layers:
            x = layer(x, x_pos, y_pos)
            
        # 3. Global average pooling over spatial sequence
        x_gap = x.mean(dim=1) # (B, d_e)
        
        # 4. Compute logits using unit-normalised classifier weight (scaled by s_F)
        w_bar = rss(self.classifier_weight) # (num_classes, d_e)
        logits = torch.exp(self.s_F) * F.linear(x_gap, w_bar)
        
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())