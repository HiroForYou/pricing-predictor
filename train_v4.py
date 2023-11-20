import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMModel
from utils import prepare_data

# Cargar los datos desde el archivo CSV
data = pd.read_csv('./data/2021-2023_jun_filtered_data.csv')

# Preparar los datos
X, y = prepare_data(data)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Convertir los datos a tensores de PyTorch
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).view(-1, 1).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).view(-1, 1).float()

# Crear conjuntos de datos y DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Definir el modelo LSTM
input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = 1
num_layers = 2

model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento del modelo
num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Evaluación del modelo
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        test_outputs = model(inputs)
        test_loss += criterion(test_outputs, targets).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Average Loss on test set: {avg_test_loss}')

# Guardar los pesos del modelo
folder_name = 'weights'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

current_time = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = f"{folder_name}/{current_time}-avg-test-loss-{avg_test_loss:.4f}.pth"
torch.save(model.state_dict(), file_name)
