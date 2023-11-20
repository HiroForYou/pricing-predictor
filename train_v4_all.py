import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from models import ImprovedLSTMModel
from utils import (
    plot_losses,
    DataProcessor,
    ModelTrainer,
    DataLoaderManager,
    ModelSaver,
)

# Verificar si CUDA está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    data_processor = DataProcessor(data_path="./data/homogeneo.csv")
    data = data_processor.load_data()

    X, y = data_processor.prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).view(-1, 1).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).view(-1, 1).float()

    data_loader_manager = DataLoaderManager(device=device)
    train_loader, test_loader = data_loader_manager.create_data_loaders(
        X_train, y_train, X_test, y_test
    )

    input_dim = X_train.shape[1]
    hidden_dim = 32
    output_dim = 1
    num_layers = 2

    model = ImprovedLSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = ModelTrainer(model, criterion, optimizer, train_loader, test_loader)
    trainer.train_model()

    avg_test_loss = trainer.evaluate_model()

    model_saver = ModelSaver(model)
    model_saver.save_model_weights(avg_test_loss)

    train_losses = trainer.get_train_losses()
    test_losses = trainer.get_test_losses()

    plot_losses(train_losses, test_losses)


if __name__ == "__main__":
    main()
