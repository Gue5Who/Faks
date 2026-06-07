from train import Training
from test import Testing
from model import ModelCT
import json
import os

if __name__ == '__main__':

     main_path_to_data = "//home/sumakj/PSUF/5nal/5-naloga/processed" # mapa v kateri se nahajajo vsi procesirani podatki
     path_to_model = "trained_models/dense_200_100_50_lr2e-4" # mapa v katero se bo shranilo: utezi naucenega modela, auc in loss med ucenjem

    # Nalozimo sezname za ucno, validacijsko in testno mnozico
    with open(os.path.join(main_path_to_data, "train_info.json")) as f:
        train_info = json.load(f)
    with open(os.path.join(main_path_to_data, "val_info.json")) as f:
        valid_info = json.load(f)
    with open(os.path.join(main_path_to_data, "test_info.json")) as f:
        test_info = json.load(f)

    # Izberemo model
    my_model = ModelCT()

    # Nastavimo hiperparametre v slovarju
    # og parametri
    #hyperparameters = {}
    #hyperparameters['learning_rate'] = 0.2e-3 # learning rate
    #hyperparameters['weight_decay'] = 0.0001 # weight decay
    #hyperparameters['total_epoch'] = 10 # total number of epochs
    #hyperparameters['multiplicator'] = 0.95 # each epoch learning rate is decreased on LR*multiplicator

    # parametri za gosto mrežo path_to_model = "trained_models/dense_200_100_50_lr2e-4"

    hyperparameters = {}
    hyperparameters["learning_rate"] = 2e-4      # lr = 2 · 10^-4
    hyperparameters["weight_decay"]  = 1e-4      # L2 regularizacija
    hyperparameters["total_epoch"]   = 30        # lahko tudi 25–30
    hyperparameters["multiplicator"] = 0.95      # LR ← LR * 0.95 na epoch



    # Ustvarimo ucni in testni razred
    TrainClass = Training(main_path_to_data)
    TestClass = Testing(main_path_to_data)

    # Naucimo model za izbrane hiperparametre
    aucs, losses = TrainClass.train(train_info, valid_info, my_model, hyperparameters, path_to_model)

    # Utezi naucenega modela
    path_to_model_weights = os.path.join(path_to_model, "trained_model_weights.pth")

    # Testiramo nas model na testni mnozici
    auc, fpr, tpr, thresholds, trues, predictions = TestClass.test(test_info, my_model, path_to_model_weights)

    print("Test set AUC result: ", auc)
