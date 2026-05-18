import torch
from multiscreen import Multiscreen

m = Multiscreen(vocab_size=10, d_e=16, n_l=2, n_h=2, d_k=8, d_v=8)
x = torch.randint(0, 10, (2, 5))
y = m(x)
print(y.shape)
