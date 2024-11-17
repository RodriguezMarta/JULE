import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

def save_processed_image(image_tensor, save_path):
    """
    Guarda una imagen procesada como archivo PNG.
    :param image_tensor: Tensor que representa la imagen.
    :param save_path: Ruta donde se guardará la imagen.
    """
    image = image_tensor.squeeze().cpu().numpy()  # Convertir de tensor a numpy array
    image = Image.fromarray(image.astype('uint8'))  # Convertir de numpy array a imagen
    image.save(save_path)
def save_labels(labels, save_path):
    """
    Guarda las etiquetas de las imágenes en un archivo de texto.
    :param labels: Etiquetas de las imágenes.
    :param save_path: Ruta donde se guardarán las etiquetas.
    """
    with open(save_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def label_to_onehot(labels, num_classes):
    """Labels to one-hot format"""
    onehot = np.zeros((len(labels), num_classes))
    onehot[np.arange(len(labels)), labels] = 1
    return onehot

def plot_image_with_labels(image, labels, class_names):
    """Visualize image with label"""
    plt.imshow(image, cmap='gray')
    plt.title(f'Labels: {", ".join([class_names[i] for i in range(len(class_names)) if labels[i] == 1])}')
    plt.show()

def calculate_metrics(y_true, y_pred):
    """Get metrics"""
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

