import numpy as np
import torch
from torch.utils.data import DataLoader
from datareader import DataReader
from sklearn.metrics import roc_auc_score, roc_curve

class Testing():
    def __init__(self, main_path_to_data):
        self.main_path_to_data = main_path_to_data

    def test(self, test_info, model, path_to_model_weights):
        """Function for testing the model on test_info.

        Args:
            test_info (list): list of paths to 10 central slices per patient (ordered)
            model (nn.Module): architecture of the model
            path_to_model_weights (string): absolute path to the model weights

        Returns:
            auc (float): AUC calculated on test set
            fpr (ndarray): increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
            tpr (ndarray): increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i]
            thresholds (ndarray): decreasing thresholds on the decision function used to compute fpr and tpr
            trues (list): ground truth labels (0/1)
            predictions (list): predictions [0,1]
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1. Load trained model and set it to eval mode
        model.to(device)
        model.load_state_dict(torch.load(path_to_model_weights))
        model.eval()

        # 2. Create datalodaer
        test_datareader = DataReader(self.main_path_to_data, test_info)
        test_generator = DataLoader(test_datareader, batch_size=10, shuffle=False, pin_memory = True, num_workers=2)

        # 3. Calculate metrics
        predictions = []
        trues = []

        for item_test in test_generator:
            # Load images (x) and labels (y)
            x, y = item_test
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            with torch.no_grad():
                y_hat = model.forward(x)
                y_hat = torch.sigmoid(y_hat) # In training we are using BCEWithLogitsLoss for improved performance (sigmoid is already embedded), here we have to add it
            predictions.append(np.mean(y_hat.cpu().numpy()))
            trues.append(y.cpu().numpy()[0])

        auc = roc_auc_score(trues, predictions)
        fpr, tpr, thresholds = roc_curve(trues, predictions, pos_label=1)

        return auc, fpr, tpr, thresholds, trues, predictions
