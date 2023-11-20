import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo (puedes reemplazar esto con tus propios datos)
np.random.seed(42)
data = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)
data = data.astype(np.float32)

# Normalización de datos
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Función para crear secuencias de entrada y salida
def create_sequences(data, seq_length, output_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - output_length + 1):
        seq = data[i:i + seq_length]
        label = data[i + seq_length:i + seq_length + output_length]
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)

# Hiperparámetros
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 6
num_epochs = 100
learning_rate = 0.01
sequence_length = 10

# Crear secuencias de entrada y salida
sequences, targets = create_sequences(data, sequence_length, output_size)

# Convertir a tensores de PyTorch
sequences = torch.from_numpy(sequences).unsqueeze(dim=2)
targets = torch.from_numpy(targets)

# Definir el modelo LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Instanciar el modelo
model = LSTM(input_size, hidden_size, num_layers, output_size)

# Función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento del modelo
for epoch in range(num_epochs):
    outputs = model(sequences)
    optimizer.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluar el modelo
model.eval()
with torch.no_grad():
    test_sequence = data[-sequence_length:]
    test_sequence = torch.from_numpy(test_sequence).view(1, sequence_length, 1).float()
    predicted_values = model(test_sequence).numpy().flatten()

# Visualizar resultados
plt.plot(data, label='Datos reales')
plt.axvline(x=len(data) - sequence_length, color='r', linestyle='--', label='Fin de datos de entrenamiento')
plt.plot(np.arange(len(data), len(data) + output_size), predicted_values, marker='o', color='g', label='Valores pronosticados')
plt.legend()
plt.show()
