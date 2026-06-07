import json
import torch
from model import ModelCT
from datareader import DataReader
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
import os

# Razred za shranjevanje konvolucij
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nalozimo testne podatke
    main_path_to_data = "/data/PSUF_naloge/5-naloga/processed"
    model_folder = "trained_models/testrun123"
    with open (os.path.join(main_path_to_data, "test_info.json")) as fp:
            test_info = json.load(fp)

    # Nalozimo model, ki ga damo v eval mode
    model = ModelCT()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder, "trained_model_weights.pth")))
    model.eval()

    # Inicializiramo razred v SO in registriramo kavlje v nasem modelu
    SO = SaveOutput()
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            handle = layer.register_forward_hook(SO)

    # Naredimo test_generator z batch_size=1
    test_datareader = DataReader(main_path_to_data, test_info)
    test_generator = DataLoader(test_datareader, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

    # Vzamemo prvi testni primer npr.
    item_test = next(iter(test_generator))
    # Propagiramo prvi testni primer skozi mrezo, razred SaveOutput si shrani vse konvolucije
    input_image = item_test[0].to(device)
    _ = model(input_image)

    # Izberemo katero konvolucijo bi radi pogledali (color_channel bo vedno 0)
    color_channel = 0 # indeks barvnega kanala (pri nas le 1 kanal)
    idx_layer = 5 # indeks konvolucijske plasti (Conv2d) - (pri nas 21)
    idx_convolution = 17 # indeks konvolucije na dani plasti (max odvisen od plasti)

    # Vizualiziramo
    convolution_0_5_17 = SO.outputs[idx_layer][color_channel][idx_convolution][:,:].cpu().detach().numpy()
    plt.figure()
    plt.imshow(np.rot90(convolution_0_5_17), cmap="gray")

    plt.figure()
    plt.imshow(np.rot90(input_image.cpu().numpy()[0,0,:,:]),cmap="gray")
