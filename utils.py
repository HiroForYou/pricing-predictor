import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
import matplotlib.pyplot as plt
import torch

# Mapeo de nombres de meses a sus formas abreviadas en inglés
months_dict = {
    "Ene": "Jan",
    "Feb": "Feb",
    "Mar": "Mar",
    "Abr": "Apr",
    "May": "May",
    "Jun": "Jun",
    "Jul": "Jul",
    "Ago": "Aug",
    "Set": "Sep",
    "Oct": "Oct",
    "Nov": "Nov",
    "Dic": "Dec",
}


def complete_missing_rows(df):
    # Crear un DataFrame con todas las combinaciones de Año y Mes posibles
    all_years = df["Año"].unique()
    all_months = list(months_dict.keys())

    all_combinations = []
    for year in all_years:
        for month in all_months:
            all_combinations.append((year, month))

    # Crear un DataFrame con todas las combinaciones y fusionarlo con el DataFrame original
    all_data = pd.DataFrame(all_combinations, columns=["Año", "Mes"])
    result = pd.merge(all_data, df, on=["Año", "Mes"], how="left")

    # Rellenar los valores faltantes en 'Costo' con 0
    result["Costo"] = result["Costo"].fillna(0)

    # Rellenar columnas vacías con valores de sus vecinos no nulos en todas las columnas
    for col in result.columns[
        2:
    ]:  # Comenzar desde la tercera columna (excluyendo 'Año' y 'Mes')
        result[col] = result[col].fillna(method="ffill")
        result[col] = result[col].fillna(method="bfill")

    return result


def sorting_df(dataframe):
    # Crear una nueva columna con la fecha combinada
    dataframe["Año-Mes"] = pd.to_datetime(
        dataframe["Año"].astype(str) + "-" + dataframe["Mes"].map(months_dict),
        format="%Y-%b",
    )

    # Ordenar el DataFrame según la nueva columna de fecha
    dataframe = dataframe.sort_values("Año-Mes")

    # Eliminar la columna temporal 'Año-Mes'
    dataframe = dataframe.drop("Año-Mes", axis=1)

    return dataframe


class ModelSaver:
    def __init__(self, model, folder_name="weights"):
        self.model = model.to('cpu')
        self.folder_name = folder_name

    def save_model_weights(self, avg_test_loss):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = (
            f"{self.folder_name}/{current_time}-avg-test-loss-{avg_test_loss:.4f}.pth"
        )
        torch.save(self.model.state_dict(), file_name)

class DataLoaderManager:
    def __init__(self, batch_size=128, device='cpu'):
        self.device = device
        self.batch_size = batch_size

    def create_data_loaders(self, X_train, y_train, X_test, y_test):
        # Mover los datos a la GPU si está disponible
        train_dataset = TensorDataset(X_train.to(self.device), y_train.to(self.device))
        test_dataset = TensorDataset(X_test.to(self.device), y_test.to(self.device))

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return train_loader, test_loader

class ModelTrainer:
    def __init__(
        self, model, criterion, optimizer, train_loader, test_loader, num_epochs=4000
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.train_losses = []
        self.test_losses = []

    def train_model(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            for inputs, targets in self.train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                test_loss = self.evaluate_model()
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {loss.item()}, Test Loss: {test_loss}"
                )

                self.train_losses.append(loss.item())
                self.test_losses.append(test_loss)

    def evaluate_model(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                test_outputs = self.model(inputs)
                test_loss += self.criterion(test_outputs, targets).item()

            avg_test_loss = test_loss / len(self.test_loader)
            return avg_test_loss

    def get_train_losses(self):
        return self.train_losses

    def get_test_losses(self):
        return self.test_losses


class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        return pd.read_csv(self.data_path)

    def prepare_data(self, df):
        label_encoders = {}
        categorical_columns = [
            "Año",
            "Mes",
            "Marca",
            "Genero",
            "Familia Agrupada",
            "Categoria",
            "Sub categoria",
            "Nivel 6",
            "Ingresos",
            "Precio",
        ]

        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

        scaler = StandardScaler()
        df["Costo"] = scaler.fit_transform(df["Costo"].values.reshape(-1, 1))

        # Guardar los label encoders y el scaler
        for column in categorical_columns:
            dump(label_encoders[column], f"./encoders/label_encoder_{column}.joblib")
        dump(scaler, "./encoders/standard_scaler.joblib")

        X = df.drop("Costo", axis=1).values
        y = df["Costo"].values

        return X, y


def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.title("Training and Test Losses over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./images/loss_plot.png")
    plt.show()
