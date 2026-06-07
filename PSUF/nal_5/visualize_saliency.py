import torch
import numpy as np
from captum.attr import Saliency
from datareader import DataReader
from model import ModelCT
from torch.utils.data import DataLoader
import os
import json
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

if __name__ == "__main__":
    # Nalozimo testne podatke
    main_path_to_data = "/data/PSUF_naloge/5-naloga/processed"
    model_folder = "trained_models/testrun123"

    with open (os.path.join(main_path_to_data, "test_info.json")) as fp:
            test_info = json.load(fp)

    # Nalozimo model, ga damo v eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelCT()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder, "trained_model_weights.pth")))
    model.eval()

    # Naredimo testni generator
    test_datareader = DataReader(main_path_to_data, test_info)
    test_generator = DataLoader(test_datareader, batch_size=1, shuffle=False, pin_memory = True, num_workers=2)

    # Iz knjiznice captum nalozimo Saliency
    saliency = Saliency(model)

    # V testnih podatkih poiscemo primer z dobro klasifikacijo hude okuzbe (y==1, y_hat > 0.95)
    for item_test in test_generator:

        x, y = item_test
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        y_hat = model.forward(x)
        y_hat = torch.sigmoid(y_hat)

        if int(y) == 1 and float(y_hat) > 0.95:
            attribution = saliency.attribute(x)
            attribution = np.rot90(attribution.detach().cpu().numpy().squeeze())
            original = np.rot90(x.cpu().numpy().squeeze())

            # Po zelji zgladimo saliency z gaussom
            attribution = gaussian_filter(attribution, sigma=2)

            # Vizualiziramo saliency map in originalno sliko
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_title("Orignal", fontweight='bold', fontsize=10)
            ax1.imshow(original, cmap='Greys_r')
            ax1.axis('off')

            ax2.set_title("Relativni doprinos pikslov",  fontweight='bold', fontsize=10)
            ax2.imshow(original, cmap='gray')
            ax2.imshow(np.ma.masked_where(attribution==0, attribution), alpha=0.8, cmap='RdBu')
            ax2.axis('off')

            break
