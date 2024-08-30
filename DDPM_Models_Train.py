from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
import os
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    EnsureChannelFirstd,
    CenterSpatialCropd,
    Compose,
    Lambdad,
    LoadImaged,
    Resized,
    ScaleIntensityd,
)

from torch.cuda.amp import GradScaler, autocast
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet, VQVAE
from monai.utils import set_determinism
import glob
from tqdm import tqdm
print_config()




ls = glob.glob(r"D:\Dataset_NonCA\Internal_Risk\*")
ls_nii_in = [os.path.join(i,"PRE.nii.gz") for i in ls]
ls_nii_out = [os.path.join(i,"HBP.nii.gz") for i in ls]

train_datalist = [{"image": pre,"target": target} for pre,target in zip(ls_nii_in,ls_nii_out)]



data_transform = Compose(
    [
        LoadImaged(keys=["image","target"]),
        EnsureChannelFirstd(keys=["image","target"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image","target"])


    ]
)


# %%
train_ds = CacheDataset(data=train_datalist, transform=data_transform)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")








vqvae_image = VQVAE(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256),
    num_res_channels=256,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    num_embeddings=16384,
    embedding_dim=3,
    output_act="relu"
)


vqvae_target = VQVAE(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256),
    num_res_channels=256,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    num_embeddings=16384,
    embedding_dim=3,
    output_act="relu"
)


vqvae_image.to(device)
vqvae_target.to(device)





with torch.no_grad():
    x = next(iter(train_loader))["image"].cuda()
    z = vqvae_image.encode_stage_2_inputs(x)
    x_re,_ = vqvae_image(x)
    plt.subplots(1, 2, figsize=(10, 6), dpi=300)
    plt.subplot(1, 2, 1)
    plt.imshow(x[0][0][:,:,15].detach().cpu().numpy().T,"gray")
    plt.subplot(1, 2, 2)
    plt.imshow(x_re[0][0][:,:,15].detach().cpu().numpy().T,"gray")
    plt.show()





device = torch.device("cuda")





model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=6,
    out_channels=3,
    num_channels=(128, 256, 512),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=(0,256,512),
    with_conditioning=False,
)


model.to(device)






scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
inferer = DiffusionInferer(scheduler)



n_epochs = 2000
val_interval = 50
epoch_loss_list = []
val_epoch_loss_list = []


scaler = GradScaler()


out_folder = r"your path"
ckpt_name = "P2AP" #model name
ckpt_path = os.path.join(out_folder,ckpt_name)
if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)

model_name = "P2AP.pt"
img_name = "P2AP.png"






for epoch in range(n_epochs):
    model.train()

    epoch_loss = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        with torch.no_grad():
            vqvae_image.eval()
            vqvae_target.eval()
            x_lat = vqvae_image.encode_stage_2_inputs(images)
            y_lat = vqvae_target.encode_stage_2_inputs(targets)


        optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (len(images),)).to(device)  # pick a random time step t

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(y_lat).to(device)
            noisy_y_lat = scheduler.add_noise(
                original_samples=y_lat, noise=noise, timesteps=timesteps
            )
            combined = torch.cat(
                (x_lat, noisy_y_lat), dim=1
            )
            prediction = model(x=combined, timesteps=timesteps)
            # Get model prediction
            loss = F.mse_loss(prediction.float(), noise.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))


    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(ckpt_path, str(epoch + 1) + model_name))

