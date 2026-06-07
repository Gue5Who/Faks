import numpy as np
import torch
from torch.utils.data import DataLoader
from datareader import DataReader
from sklearn.metrics import roc_auc_score
import time
import os

class Training():
    def __init__(self, main_path_to_data):
        self.main_path_to_data = main_path_to_data

    def train(self, train_info, valid_info, model, hyperparameters, path_to_model):
        """Function for training the model on train_info.

        Args:
            train_info (list): list of paths to one central slice per patient (shuffled)
            valid_info (list): list of paths to 10 central slices per patient (ordered)
            model (nn.Module): architecture of the model
            hyperparameters (dictionary): dictionary of hyperparameters (learning rate, weight decay, multiplicator)
            path_to_model (string): absolute path to the folder where outputs will be saved

        Returns:
            aucs (ndarray): array of AUCs during validation (one AUC per epoch)
            losses (ndarray): array of LOSSes during training (one running loss per epoch)
        """
        # 0. Check which device is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Create folder to save the model weights, aucs and losses
        try:
            os.mkdir(path_to_model)
        except: # folder already exists, add current time to avoid duplication
            os.mkdir(path_to_model + str(int(time.time())))

        # 2. Load hyperparameters
        learning_rate = hyperparameters['learning_rate']
        weight_decay = hyperparameters['weight_decay']
        total_epoch = hyperparameters['total_epoch']
        multiplicator = hyperparameters['multiplicator']

        # 4. Create train and validation generators, batch_size = 10 for validation generator (10 central slices)
        train_datareader = DataReader(self.main_path_to_data, train_info)
        train_generator = DataLoader(train_datareader, batch_size=16, shuffle=True, pin_memory=True, num_workers=2)

        valid_datareader = DataReader(self.main_path_to_data, valid_info)
        valid_generator = DataLoader(valid_datareader, batch_size=10, shuffle=False, pin_memory=True, num_workers=2)

        # 5. Move model to the available device
        model.to(device)

        # 6. Define criterion function, optimizer and scheduler
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, multiplicator, last_epoch=-1)

        # 7. Creat lists for tracking AUC and Losses during training
        aucs = []
        losses = []
        best_auc = -np.inf

        # 8. Run training
        for epoch in range(total_epoch):
            start = time.time()
            print('Epoch: %d/%d' % (epoch + 1, total_epoch))

            running_loss = 0
            # A) Train model
            model.train()  # put model in training mode
            for item_train in train_generator:
                # Load images (x) and labels (y)
                x, y = item_train
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                optimizer.zero_grad()
                y_hat = model.forward(x)
                loss = criterion(y_hat, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track loss change
                running_loss += loss.item()

            # B) Validate model
            predictions = []
            trues = []

            model.eval() # put model in eval mode
            for item_valid in valid_generator:
                # Load images (x) and labels (y)
                x, y = item_valid
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                with torch.no_grad():
                    y_hat = model.forward(x)
                    y_hat = torch.sigmoid(y_hat) # In training we are using BCEWithLogitsLoss for improved performance (sigmoid is already embedded), here we have to add it

                predictions.append(np.mean(y_hat.cpu().numpy())) # Calculate mean of 10 predictions
                trues.append(int(y.cpu().numpy()[0]))

            auc = roc_auc_score(trues, predictions)

            # C) Track changes, update LR, save best model
            print("AUC: ", auc, ", Running loss: ", running_loss/len(train_generator), ", Time: ", time.time()-start)

            if auc > best_auc:
                torch.save(model.state_dict(), os.path.join(path_to_model, 'trained_model_weights.pth'))
                best_auc = auc


            aucs.append(auc)
            losses.append(running_loss/len(train_generator))
            scheduler.step()

        np.save(os.path.join(path_to_model, 'AUCS.npy'), np.array(aucs))
        np.save(os.path.join(path_to_model, 'LOSSES.npy'), np.array(losses))

        return aucs, losses
