from model import ModelCT
from matplotlib import pyplot as plt
import numpy as np
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_folder = "trained_models/testrun123" # mapa naucenega modela
model = ModelCT()
model.to(device)
model.load_state_dict(torch.load(os.path.join(model_folder, "trained_model_weights.pth")))
model.eval()
# Nalozimo npr. filtre iz prve konvolucijske plasti backbone.conv1
weights = model.backbone.conv1.weight.data.cpu().numpy()
# weights.shape = (64,1,7,7) -> 64 filtrov, z 1 kanalom, velikosti 7x7

# Vizualizacija 21. filtra iz prve plasti (backbone.conv1)
plt.imshow(weights[20,0,:,:], cmap='gray')
