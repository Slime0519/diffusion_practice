import argparse
import copy

import pytorch_lightning as pl
import torch
import torchvision
import wandb

from pytorch_lightning.loggers import WandbLogger
from torch import nn
from diffusion_clone.diffusion_process import Diffusion
from modules import UNet
from utils import get_data
from PIL import Image
def tensor2image(images):
    grid = torchvision.utils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    return ndarr

class PlDiffusionModule(pl.LightningModule):
    def __init__(self, model, diffusion: Diffusion):
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

    def training_epoch_end(self, losses) -> None:
        avg_loss = torch.stack([x['loss'] for x in losses]).mean()
        # self.log('train_loss', avg_loss)
        self.log('train_loss', avg_loss)
        print(avg_loss)

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch[0])
        return loss

    def validation_epoch_end(self, losses):
        avg_loss = torch.stack(losses).mean()
        # self.log('train_loss', avg_loss)
        self.log('val_loss', avg_loss)
        sampled_imgs = self.diffusion.sample(self.model, self.trainer.val_dataloaders[0].batch_size)
        img = tensor2image(sampled_imgs)
        self.trainer.logger.experiment.log({'samples' : [wandb.Image(img)]})
        print(avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return optimizer


def main():
    device = 'cuda'
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 2
    args.image_size = 64
    args.dataset_path = r"C:\Users\VISAI_JMLEE\Desktop\workspace\diffusion_practice\archive"
    args.device = "cuda"
    args.lr = 3e-4
    unet = UNet()
    diffusion = Diffusion(img_size=64, device=device, noise_steps=10)
    model = PlDiffusionModule(unet, diffusion)
    dataloader = get_data(args)

    logdir = r"C:\Users\VISAI_JMLEE\Desktop\workspace\diffusion_practice\wandb_log"
    logger = WandbLogger(project='diffusion', name='initial_logging_test', log_model='all', save_dir=logdir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='train_loss', mode='min')
    trainer = pl.Trainer(gpus=1,
                         max_epochs=500,
                         precision=32,
                         accelerator='gpu',
                         # limit_train_batches=4,
                         limit_val_batches=2,
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         )
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=copy.copy(dataloader))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    main()
