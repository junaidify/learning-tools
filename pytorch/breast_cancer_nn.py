import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
# print(df.head())
# print(df.shape)

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Label Encoding

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Numpy arrays to PyTorch tensors

x_train_tensor = torch.from_numpy(x_train)
x_test_tensor = torch.from_numpy(x_test)
y_train_tensor = torch.from_numpy(y_train).reshape(-1, 1)
y_test_tensor = torch.from_numpy(y_test).reshape(-1, 1)


# Define the model

class MySimpleNN(): 
    def __init__(self, X) -> None:
        self.weights = torch.rand(X.shape[1], 1, dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)


    def forward(self, X): 
        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred

    def loss_function(self, y_pred, y): 
        # Clamp predictions to avoid log(0)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)


        #Calculate loss
        loss = -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()
        return loss


learning_rate = 0.1
epochs = 25

# Training Pipeline

# create Model
model = MySimpleNN(x_train_tensor)

# define loop 
for epoch in range(epochs): 
    # forward pass
    y_pred = model.forward(x_train_tensor)

    # loss calculate
    loss = model.loss_function(y_pred, y_train_tensor)

    # backward pass
    loss.backward()

    #parameters update
    with torch.no_grad(): 
        model.weights.data -= learning_rate * model.weights.grad
        model.bias.data -=learning_rate * model.bias.grad

    #zero gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    #print loss in each epoch
    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    # model.bias

    ## Evaluation

with torch.no_grad(): 
    y_pred = model.forward(x_test_tensor)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred == y_test_tensor).float().mean()
    print(f"Accuracy: {accuracy.item()}")
