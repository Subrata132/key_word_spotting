import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_loader import load_data
from model import CNNModelTiny
from constants import TrainingParams
from utils import create_folder, gpu_to_cpu


def train_epoch(model, data_generator, optimizer, criterion, device, epoch, max_epoch, batch_size):
    model.train()
    train_loss = 0
    nb_train_batches = 0
    i = 0
    len_dataloader = len(data_generator)
    for x, y in tqdm(data_generator, desc=f'Epoch: {epoch+1}/{max_epoch}'):
        p = float(i + epoch * len_dataloader) / max_epoch / len_dataloader
        x = torch.tensor(x).to(device).float()
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
        optimizer.zero_grad()
        pred_y = model(x)
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        nb_train_batches += 1
        i += 1
    train_loss /= nb_train_batches
    return train_loss


def test_epoch(model, data_generator, criterion, device, desc):
    model.eval()
    test_loss = 0
    nb_test_batches = 0
    all_true_label, all_pred_label = [], []
    with torch.no_grad():
        for x, y in tqdm(data_generator, desc=desc):
            x = torch.tensor(x).to(device).float()
            y = torch.tensor(y).to(device)
            pred_y = model(x)
            loss = criterion(pred_y, y)
            test_loss += loss.item()
            nb_test_batches += 1
            y = gpu_to_cpu(y)
            pred_y = gpu_to_cpu(pred_y)
            all_true_label += list(y)
            all_pred_label += list(np.argmax(pred_y, axis=1))
    test_loss /= nb_test_batches
    accuracy = accuracy_score(all_true_label, all_pred_label)
    return test_loss, accuracy


def train():
    training_params = TrainingParams()
    create_folder(training_params.save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNNModelTiny().to(device=device)
    # summary(model, [(1, 513, 1034), (0.)])
    train_loader = load_data(
        batch_size=training_params.batch_size,
        feat_path=training_params.feat_path,
        csv_path=training_params.train_csv_loc,
        shuffle=True
    )
    val_loader = load_data(
        batch_size=training_params.batch_size,
        feat_path=training_params.feat_path,
        csv_path=training_params.val_csv_loc,
        shuffle=False
    )
    test_loader = load_data(
        batch_size=training_params.batch_size,
        feat_path=training_params.feat_path,
        csv_path=training_params.test_csv_loc,
        shuffle=False
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params.lr)
    best_val_epoch, best_val_loss = 0, 1e10
    for epoch in range(training_params.max_epoch):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            training_params.max_epoch, training_params.batch_size
        )
        val_loss, val_accuracy = test_epoch(
            model, val_loader, criterion, device, "Validating: "
        )
        print(f'Training Loss: {train_loss} | Validation Loss: {val_loss} | Val Acc: {val_accuracy} ')
        if val_loss <= best_val_loss:
            torch.save(model.state_dict(), os.path.join(training_params.save_path, training_params.model_name))
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
    print(f'Best model saved at {best_val_epoch}th epoch!')
    model.load_state_dict(torch.load(os.path.join(training_params.save_path, training_params.model_name), map_location='cpu'))
    print('Loading unseen test dataset:')
    test_loss, test_accuracy = test_epoch(model, test_loader, criterion, device, "Testing: ")
    print(f'Test Loss: {test_loss} | Test Acc: {test_accuracy}')
