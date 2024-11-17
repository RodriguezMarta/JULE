import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, save_dir="models/"):
    """
    Entrena el modelo y evalúa en el conjunto de validación.
    :param model: Modelo PyTorch
    :param train_loader: DataLoader para entrenamiento
    :param val_loader: DataLoader para validación
    :param optimizer: Optimizador
    :param criterion: Función de pérdida
    :param device: Dispositivo (CPU o GPU)
    :param epochs: Número de épocas
    :param save_dir: Carpeta para guardar checkpoints
    """
    # Asegurarse de que la carpeta para guardar el modelo existe
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Bucle de entrenamiento
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings, cluster_probs = model(images)
            loss = criterion(embeddings, cluster_probs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                embeddings, cluster_probs = model(images)
                loss = criterion(embeddings, cluster_probs, labels)
                val_loss += loss.item()

        # Promedios de pérdida
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"Model saved at epoch {epoch + 1}")

    print("Training complete.")
