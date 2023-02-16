import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from modules import UNet
from utils import get_data


class Diffusion:
    # diffusion process with N x N images
    def __init__(self, img_size=256, noise_steps=1000, beta_range=[1e-4, 0.02], device="cuda"):
        self.alpha = 1
        self.img_size = img_size
        self.noise_steps = noise_steps
        self.device = device
        self.beta_range = beta_range

        self.beta = self.get_beta()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)

    def get_beta(self):
        return torch.linspace(self.beta_range[0], self.beta_range[1], self.noise_steps).to(self.device)

    def get_noise(self, x, t):
        # add noise to image x at timestep t
        mean, std = torch.sqrt(self.alpha_hat[t][:, None, None, None]) * x, \
            torch.sqrt(1 - self.alpha_hat[t][:, None, None, None])
        z = torch.randn_like(x).to(self.device)
        noise = mean + std * z
        return noise, z

    def sample(self, model: nn.Module, n):
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                unsqueeze = lambda x: x[:, None, None, None]
                alpha, alpha_hat, beta = unsqueeze(self.alpha[t]), unsqueeze(self.alpha_hat[t]), unsqueeze(self.beta[t])

                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def get_loss(x, diffusion: Diffusion, model, loss_fn):
    t = diffusion.sample_timesteps(x.shape[0])
    diffusion.get_noise(x, t)
    x_t, noise = diffusion.get_noise(x, t)
    predicted_noise = model(x_t, t)
    loss = loss_fn(predicted_noise, noise)
    return loss


def train(args):
    # train the model
    dataloader = get_data(args)
    diffusion = Diffusion(img_size=64, device='cuda')
    model = UNet().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to('cuda')
            loss = get_loss(images, diffusion, model, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

        # model.eval()
        # sampled_images = diffusion.sample(model, images.shape[0])


def train_test():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 8
    args.image_size = 64
    args.dataset_path = r"C:\Users\User\Desktop\workspace\Diffusion-Models-pytorch\archive"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    diffusion = Diffusion(noise_steps=10)
    # noise test
    # x = np.zeros((256, 256, 3))
    # noisy_x = diffusion.get_noise(x, 9)[0]
    train_test()
    #
    # cv2.imshow("", noisy_x)
    # cv2.waitKey(0)
