import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datetime import datetime as dt


class imdb_Dataset(Dataset):
    def __init__(self, feats, targs):
        self.X = torch.tensor(feats).float()
        self.y = torch.tensor(targs).float()

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        x = self.X[i]

        return x, self.y[i]
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 1)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss

def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()
    

    avg_loss = total_loss / num_batches
    return avg_loss

def predict(data_loader, model):
    model = model
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    
    return output

def get_pred_df(target_col, test_loader, y_test, model, batch_size=32):
    
    """Create dataframe with actual and un-standardized predicted return values """

    preds = predict(test_loader, model=model)
    preds_actual = (preds * target_col.std()) + target_col.mean()
    
    y_test_actual = (y_test * target_col.std()) + target_col.std()
    
    array = np.concatenate([preds_actual, y_test_actual], axis=1)
    
    df_out = pd.DataFrame(array, columns=['Profit_predicted', 'Profit_actual'])
    df_out['squared_error'] = (df_out['Profit_predicted'] - df_out['Profit_actual'])**2
    
   

    return df_out

def plot_losses(train_losses, test_losses):
    x = range(len(train_losses))
    plt.plot(x, train_losses, label='Train Loss')
    plt.plot(x, test_losses, label='Test Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.show()
    
