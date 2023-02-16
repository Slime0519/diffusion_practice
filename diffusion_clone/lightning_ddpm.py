import argparse

import pytorch_lightning as pl
import torch
from torch import nn

from diffusion_clone.diffusion_process import Diffusion
from modules import UNet
from utils import get_data


class PlDiffusionModule(pl.LightningModule):
    def __init__(self, model, diffusion : Diffusion):
        super().__init__()

        self.model = UNet()
        self.diffusion = diffusion
        self.loss_fn = nn.MSELoss()

    def forward(self, x, t):
        return self.model(x, t)

    def _get_loss(self, x):
        t = self.diffusion.sample_timesteps(x.shape[0])
        x_t, noise = self.diffusion.get_noise(x, t)
        predicted_noise = self.model(x_t, t)
        loss = self.loss_fn(predicted_noise, noise)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return optimizer


def main():
    device = 'cuda'
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 8
    args.image_size = 64
    args.dataset_path = r"C:\Users\User\Desktop\workspace\Diffusion-Models-pytorch\archive"
    args.device = "cuda"
    args.lr = 3e-4
    unet = UNet()
    diffusion = Diffusion(img_size=64, device=device)
    model = PlDiffusionModule(unet, diffusion)
    dataloader = get_data(args)

    trainer = pl.Trainer(gpus=1, max_epochs=500, precision=16, accelerator='gpu')
    trainer.fit(model, train_dataloaders=dataloader)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    main()