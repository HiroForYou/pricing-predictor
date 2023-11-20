import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

from models import ImprovedLSTMModel

matplotlib.use("Qt5Agg")
# Cargar los datos de prueba
MODEL_PATH = "./weights/2023-11-16_15-54-54-avg-test-loss-0.9131.pth"
data = pd.read_csv("./data/unit_homogeneo.csv")
data_copy = data.copy()
standar_scaler_file = "./encoders/standard_scaler.joblib"
standar_scaler = joblib.load(standar_scaler_file)


# Preparar los datos para la predicción
def prepare_data_for_prediction(df):
    # Cargar los LabelEncoders previamente guardados
    categorical_columns = [
        "Año",
        "Mes",
        "Marca",
        "Genero",
        "Familia Agrupada",
        "Categoria",
        "Sub categoria",
        "Nivel 6",
        #"Ingresos",
        "Precio",
    ]
    label_encoders = {}

    for column in categorical_columns:
        # Carga los LabelEncoders guardados
        label_encoder_file = f"./encoders/label_encoder_{column}.joblib"
        label_encoders[column] = joblib.load(label_encoder_file)

        # Aplica la transformación a los datos de prueba
        df[column] = label_encoders[column].transform(df[column])

    df["Costo"] = standar_scaler.transform(df["Costo"].values.reshape(-1, 1))
    X = df.drop("Costo", axis=1).values
    y = df["Costo"].values

    return X, y


def getStandarTarget(df):
    df["Costo"] = standar_scaler.transform(df["Costo"].values.reshape(-1, 1))
    y = df["Costo"].values
    return y


# Preparar los datos de prueba
X_test, y_test = prepare_data_for_prediction(data)

# Convertir los datos a tensores de PyTorch
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).view(-1, 1).float()

# Crear un DataLoader para los datos de prueba
test_dataset = TensorDataset(X_test, y_test)
batch_size = 1  # Puedes cambiar el tamaño del lote según sea necesario
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Asegúrate de inicializar los parámetros del modelo correctamente
input_dim = X_test.shape[1]
hidden_dim = 32
output_dim = 1
num_layers = 2

# Cargar el modelo entrenado
model = ImprovedLSTMModel(input_dim, hidden_dim, output_dim, num_layers)

# Cargar los pesos del modelo entrenado
model.load_state_dict(torch.load(MODEL_PATH))

# Modo de evaluación
model.eval()

# Realizar predicciones
predictions = []
# targets = []
with torch.no_grad():
    for inputs, target in test_loader:
        output = model(inputs)
        predictions.append(output.item())
        # targets.append(target.item())

# Graficar 'Año'-'Mes' vs 'Costo' (Valores predichos)
year_month = data_copy.apply(lambda row: f"{row['Año']}-{row['Mes']}", axis=1).tolist()
targets = getStandarTarget(data_copy)
plt.figure(figsize=(10, 6))
plt.plot(year_month, predictions, label="Predicciones", marker="o", color="blue")
plt.plot(year_month, targets, label="Target", marker="o", color="red")
plt.xlabel("Año-Mes")
plt.ylabel("Costo Predicho")
plt.title("Predicción de Costo por Año-Mes")
plt.xticks(rotation=90)  # Rotar las etiquetas del eje x en 90 grados
plt.legend()
plt.savefig("./images/prediction_plot.png")
plt.tight_layout()
plt.show()
