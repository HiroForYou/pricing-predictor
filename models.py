import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Inicialización de h0 y c0 como parámetros entrenables
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))

    def forward(self, x):
        # Replicar h0 y c0 para cada elemento en el lote
        h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
        c0 = self.c0.expand(-1, x.size(0), -1).contiguous()

        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x):
        self.eval()  # Cambiar a modo de evaluación
        with torch.no_grad():
            h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
            c0 = self.c0.expand(-1, x.size(0), -1).contiguous()

            out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
            out = self.fc(out[:, -1, :])

        self.train()  # Cambiar de vuelta a modo de entrenamiento
        return out

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional=True, dropout=0.2):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Capa LSTM bidireccional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

        # Capa de normalización por lotes
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2) if bidirectional else nn.BatchNorm1d(hidden_dim)

        # Capa de dropout
        self.dropout = nn.Dropout(p=dropout)

        # Capa completamente conectada
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        # Inicialización de h0 y c0 como parámetros entrenables
        self.h0 = nn.Parameter(torch.zeros(num_layers * 2 if bidirectional else num_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(num_layers * 2 if bidirectional else num_layers, 1, hidden_dim))

    def forward(self, x):
        h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
        c0 = self.c0.expand(-1, x.size(0), -1).contiguous()

        # Capa LSTM bidireccional
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))

        # Capa de normalización por lotes
        out = self.batch_norm(out[:, -1, :])

        # Capa de dropout
        out = self.dropout(out)

        # Capa completamente conectada
        out = self.fc(out)

        return out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
            c0 = self.c0.expand(-1, x.size(0), -1).contiguous()

            out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
            out = self.batch_norm(out[:, -1, :])
            out = self.dropout(out)
            out = self.fc(out)

        self.train()
        return out
