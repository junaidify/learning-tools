import pandas as pd
from scipy.linalg import dft
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('https://drive.google.com/file/d/1ziN7gJHnr-MQb0_eFsPfPfiiDuvsmqVH/view')
X = df.drop('target', axis=1).values
Y = df['target'].values

scaler = StandardScaler()
X = scaler.fit_transform()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

X_train = torch.tensor(x_train, dtype=torch.float32)
X_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)