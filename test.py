import torch

from ddpm import Diffusion
from modules import UNet
from utils import plot_images

device = "cuda"
model = UNet().to(device)
ckpt = torch.load("unconditional_ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
x = diffusion.sample(model, n=16)
plot_images(x)