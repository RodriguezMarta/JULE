import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class JointLearningModel(nn.Module):
    def __init__(self, num_clusters=14, embedding_dim=128, pretrained=True):
        super(JointLearningModel, self).__init__()
        # Cargar ResNet50 preentrenado en ImageNet
        self.backbone = resnet50(pretrained=pretrained)
        # Modificar la capa final para extracción de características
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Reemplazamos la FC original con identidad
        # Proyección a un espacio de menor dimensión
        self.fc_embedding = nn.Linear(num_features, embedding_dim)
        # Capa de clustering
        self.clustering_layer = nn.Linear(embedding_dim, num_clusters)
    
    def forward(self, x):
        # Extraer características con ResNet50
        features = self.backbone(x)  # Salida de la backbone
        # Proyección al espacio de embeddings
        embeddings = self.fc_embedding(features)
        # Calcular probabilidades de cluster
        cluster_probs = F.softmax(self.clustering_layer(embeddings), dim=1)
        return embeddings, cluster_probs
