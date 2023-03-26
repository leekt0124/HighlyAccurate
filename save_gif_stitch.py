from hydra import core, initialize, compose
from omegaconf import OmegaConf


# CHANGE ME
DATASET_DIR = '/home/goroyeh/nuScene_dataset/media/datasets/nuscenes'
LABELS_DIR = '/home/goroyeh/nuScene_dataset/media/datasets/cvt_labels_nuscenes'


core.global_hydra.GlobalHydra.instance().clear()        # required for Hydra in notebooks

initialize(config_path='../HighlyAccurate/transformer/config')

# Add additional command line overrides
cfg = compose(
    config_name='config',
    overrides=[
        'experiment.save_dir=../logs/',                 # required for Hydra in notebooks
        #  '+experiment= gkt_nuscenes_kernel_7x1_goro_savegif', 
        'data=nuscenes',
        f'data.dataset_dir={DATASET_DIR}',
        f'data.labels_dir={LABELS_DIR}',
        'data.version=v1.0-trainval',
        'highlyaccurate.loader.batch_size=1',
    ]
)

# resolve config references
OmegaConf.resolve(cfg)

print(list(cfg.keys()))

import torch
import numpy as np

from transformer.cross_view_transformer.common import setup_data_module


# Additional splits can be added to cross_view_transformer/data/splits/nuscenes/
SPLIT = 'val_qualitative_000'
SUBSAMPLE = 10


data = setup_data_module(cfg)

dataset = data.get_split(SPLIT, loader=False)
dataset = torch.utils.data.ConcatDataset(dataset)
dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), SUBSAMPLE))

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
print(len(dataset))

from pathlib import Path
from cross_view_transformer.common import load_backbone

## -------- Packages for downloaindg file from the web ----------- ##
import aiofiles
import aiohttp
import asyncio
async def async_http_download(src_url, dest_file, chunk_size=65536):
    async with aiofiles.open(dest_file, 'wb') as fd:
        async with aiohttp.ClientSession() as session:
            async with session.get(src_url) as resp:
                async for chunk in resp.content.iter_chunked(chunk_size):
                    await fd.write(chunk)

## --------------------------------------------------------------- ##

# IPYNB:
# !mkdir -p $(dirname ${VEHICLE_CHECKPOINT_PATH})
# !wget $VEHICLE_MODEL_URL -O $VEHICLE_CHECKPOINT_PATH
# !wget $ROAD_MODEL_URL -O $ROAD_CHECKPOINT_PATH

download_vehicle_model = False
if download_vehicle_model:
    # Download a pretrained model (13 Mb)
    VEHICLE_MODEL_URL = 'https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_vehicles_50k.ckpt'
    VEHICLE_CHECKPOINT_PATH = '/home/goroyeh/GKT/segmentation/logs/cvt_nuscenes_vehicles_50k.ckpt'
    SRC_URL = VEHICLE_MODEL_URL
    DEST_FILE = VEHICLE_CHECKPOINT_PATH
    asyncio.run(async_http_download(SRC_URL, DEST_FILE))
    print(f'Successfully download VEHICLE_MODEL')
else:
    # Use our custom checkpoint path
    # Try our custom VEHICLE_MODEL_PATH
    VEHICLE_CHECKPOINT_PATH = '/home/goroyeh/GKT/segmentation/outputs/uuid_test/checkpoints/model_test.ckpt'
    print(f'Successfully load custome VEHICLE_MODEL from {VEHICLE_CHECKPOINT_PATH}')

download_road_model = True
if download_road_model:
    ROAD_MODEL_URL = 'https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_road_75k.ckpt'
    ROAD_CHECKPOINT_PATH = '/home/goroyeh/GKT/segmentation/logs/cvt_nuscenes_road_75k.ckpt'
    SRC_URL = ROAD_MODEL_URL
    DEST_FILE = ROAD_CHECKPOINT_PATH
    asyncio.run(async_http_download(SRC_URL, DEST_FILE))    
    print(f'Successfully download ROAD_MODEL')
else:
    # TODO: We should train our own ROAD_SEGMENTATION_MODEL
    # Currently our model is a model that segments VEHICLES, not ROAD
    ROAD_CHECKPOINT_PATH = '/home/goroyeh/GKT/segmentation/outputs/uuid_test_road/checkpoints/model_road.ckpt'
    print(f'Successfully load custome RODE_MODEL from {ROAD_CHECKPOINT_PATH}')

# print(f'cfg.model.encoder.backbone.pretrained_path : {cfg.model.encoder.backbone.pretrained_weights_path}')
print(f' torch.cuda.is_available(): { torch.cuda.is_available()}') # True
vehicle_network = load_backbone(VEHICLE_CHECKPOINT_PATH)
road_network = load_backbone(ROAD_CHECKPOINT_PATH)

# %load_ext autoreload
# %autoreload 2

import torch
import time
import imageio
import ipywidgets as widgets

from cross_view_transformer.visualizations.nuscenes_stitch_viz import NuScenesStitchViz


GIF_PATH = './predictions_stitch.gif'

# Show more confident predictions, note that if show_images is True, GIF quality with be degraded.
# show_images = True: the gif comes with ground-view images
viz = NuScenesStitchViz(vehicle_threshold=0.6, road_threshold=0.6, show_images=True) 

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

vehicle_network.to(device)
vehicle_network.eval()

road_network.to(device)
road_network.eval()

images = list()

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


        vehicle_pred = vehicle_network(batch)['bev']
        road_pred = road_network(batch)['bev']

        visualization = np.vstack(viz(batch, road_pred, vehicle_pred))

        images.append(visualization)


# Save a gif
duration = [0.5 for _ in images[:-1]] + [2 for _ in images[-1:]]
imageio.mimsave(GIF_PATH, images, duration=duration)
print(f'Successfully save {GIF_PATH}')