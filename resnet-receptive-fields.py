from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model = model.to(device)

"""
Embedding layer
"""

N = 50 # number neurons to randomly sample
D = 20 # size of input image
O = 10 # size of output image

# sample indices
layer = torch.randint(0, 64, (N,1))
x = torch.randint(0, O, (N,1))
y = torch.randint(0, O, (N,1))

receptive_field = torch.zeros(N, 3, D, D)
norm = torch.zeros(N)

for _ in tqdm(range(5)):

    images = torch.randn(10000, 3, D, D)
    batch_size = 32
    dataloader = DataLoader(images, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model.resnet.embedder.embedder(batch)
            activations = output[:, layer, x, y][:,:,:,None,None]
            norm += torch.squeeze(activations).sum(dim=0)
            batch = batch[:,None,:,:,:]
            receptive_field += (batch*activations).sum(dim=0).to("cpu")

receptive_field /= norm[:,None,None,None]
np.save("embedding.npy", receptive_field.numpy())

"""
Layer 1
"""

N = 50 # number neurons to randomly sample
D = 20 # size of input image
O = 10 # size of output image

# sample indices
layer = torch.randint(0, 64, (N,1))
x = torch.randint(0, O, (N,1))
y = torch.randint(0, O, (N,1))

receptive_field = torch.zeros(N, 3, D, D)
norm = torch.zeros(N)

for _ in tqdm(range(5)):

    images = torch.randn(10000, 3, D, D)
    batch_size = 32
    dataloader = DataLoader(images, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            hidden = model.resnet.embedder.embedder(batch)
            output = model.resnet.encoder.stages[0].layers[0].layer[0](hidden)
            activations = output[:, layer, x, y][:,:,:,None,None]
            norm += torch.squeeze(activations).sum(dim=0)
            batch = batch[:,None,:,:,:]
            receptive_field += (batch*activations).sum(dim=0).to("cpu")

receptive_field /= norm[:,None,None,None]
np.save("layer-1.npy", receptive_field.numpy())

"""
Layer 2
"""

N = 50 # number neurons to randomly sample
D = 20 # size of input image
O = 10 # size of output image

# sample indices
layer = torch.randint(0, 64, (N,1))
x = torch.randint(0, O, (N,1))
y = torch.randint(0, O, (N,1))

receptive_field = torch.zeros(N, 3, D, D)
norm = torch.zeros(N)

for _ in tqdm(range(1000)):

    images = torch.randn(10000, 3, D, D)
    batch_size = 32
    dataloader = DataLoader(images, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            hidden = model.resnet.embedder.embedder(batch)
            hidden2 = model.resnet.encoder.stages[0].layers[0].layer[0](hidden)
            output = model.resnet.encoder.stages[0].layers[0].layer[1](hidden2)
            activations = output[:, layer, x, y][:,:,:,None,None]
            norm += torch.squeeze(activations).sum(dim=0)
            batch = batch[:,None,:,:,:]
            receptive_field += (batch*activations).sum(dim=0).to("cpu")

receptive_field /= norm[:,None,None,None]
np.save("layer-2.npy", receptive_field.numpy())