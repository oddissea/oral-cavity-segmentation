import os
import random
import numpy as np
from torch import Tensor
import torch
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    """Configura semillas para reproducibilidad en todos los dispositivos compatibles."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Configuración para dispositivos CUDA (NVIDIA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Estas líneas se utilizan en el caso de usar cuda
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        print(f"Semilla {seed} configurada para CPU y CUDA.")
    # Configuración para dispositivos MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS no tiene opciones de determinismo equivalentes a cudnn
        print(f"Semilla {seed} configurada para CPU y MPS (Apple Silicon).")
    else:
        print(f"Semilla {seed} configurada solo para CPU.")


def get_device():
    """Detecta y devuelve el dispositivo óptimo disponible."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def calculate_iou(pred: Tensor, target: Tensor, num_classes: int) -> np.ndarray:
    """Calcula el IoU (Intersection over Union) para cada clase."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Calculamos IoU para cada clase
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        # Usamos funciones de PyTorch para operaciones lógicas con tensores
        intersection = torch.logical_and(pred_inds, target_inds).sum().item()
        union = torch.logical_or(pred_inds, target_inds).sum().item()
        iou = intersection / (union + 1e-10)
        ious.append(iou)

    return np.array(ious)


def process_batch(model, images, masks, criterion, device, num_classes):
    """
    Procesa un batch de imágenes y máscaras a través del modelo y calcula pérdidas y métricas.

    Args:
        model: Modelo de segmentación
        images: Tensor de imágenes
        masks: Tensor de máscaras
        criterion: Función de pérdida
        device: Dispositivo (CPU/CUDA/MPS)
        num_classes: Número de clases para segmentación

    Returns:
        tuple: (loss, predicciones, IoU por clase)
    """
    images = images.to(device)
    masks = masks.to(device)

    # Forward pass
    outputs = model(images)['out']
    loss = criterion(outputs, masks)

    # Calculamos métricas
    preds = torch.argmax(outputs, dim=1)
    batch_iou = calculate_iou(preds, masks, num_classes)

    return loss.item(), preds, batch_iou

def save_plot(train_data, val_data, title, ylabel, filename):
    """Guarda una gráfica comparando datos de entrenamiento y validación."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='Entrenamiento')
    plt.plot(val_data, label='Validación')
    plt.title(title)
    plt.xlabel('Época')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()