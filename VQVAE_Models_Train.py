# %%
import os
import shutil
import tempfile
import time
import glob
from monai.data import DataLoader, Dataset,CacheDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai import transforms
from monai.apps import DecathlonDataset
from monai.data import PersistentDataset
from monai.config import print_config


from torch.nn import L1Loss
from tqdm import tqdm
from monai.transforms import (
    EnsureChannelFirstd,
    CenterSpatialCropd,
    Compose,
    Lambdad,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    ScaleIntensityRangePercentilesd,
    Resized
)
from generative.networks.nets import VQVAE
print_config()



ls = glob.glob(r"D:\Dataset_NonCA\Internal_Risk\*")
ls_nii = [os.path.join(i,"AP.nii.gz") for i in ls]

train_datalist = [{"image": item} for item in ls_nii]




data_transform = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        # CenterSpatialCropd(keys=["image"], roi_size=[256, 256, 32]),
        ScaleIntensityd(keys=["image"]),

        # ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99, b_min=0, b_max=1),
        # Resized(keys=["image"], spatial_size=(128, 128, 32))

    ]
)


train_ds = CacheDataset(data=train_datalist, transform=data_transform)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)








model = VQVAE(
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









# %%


model_name = "XXX"

base_out_folder = r"your path"

out_folder = os.path.join(base_out_folder,model_name)

if not os.path.exists(out_folder):
    os.mkdir(out_folder)




optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
l1_loss = L1Loss()


n_epochs = 200
val_interval = 2
epoch_recon_loss_list = []
epoch_quant_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        # model outputs reconstruction and the quantization error
        reconstruction, quantization_loss = model(images=images)

        recons_loss = l1_loss(reconstruction.float(), images.float())

        loss = recons_loss + quantization_loss

        loss.backward()
        optimizer.step()

        epoch_loss += recons_loss.item()

        progress_bar.set_postfix(
            {"recons_loss": epoch_loss / (step + 1), "quantization_loss": quantization_loss.item() / (step + 1)}
        )
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_quant_loss_list.append(quantization_loss.item() / (step + 1))

    if (epoch + 1) % val_interval == 0:
        torch.save(model.state_dict(),os.path.join(out_folder,str(epoch+1)+model_name))

