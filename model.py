import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights


def get_model(num_classes):
    """Crea y devuelve un modelo DeepLabV3+ adaptado para el número de clases especificado."""
    # Cargamos DeepLabV3+ preentrenado en ImageNet
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)

    # Modificamos la capa de clasificación final para nuestro número de clases
    model.classifier[4] = nn.Conv2d(
        in_channels=256,
        out_channels=num_classes,
        kernel_size=1,
        stride=1
    )

    return model