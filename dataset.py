import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import List


class OralCavityDataset(Dataset):
    def __init__(self, img_paths: List[str], transform=None):
        self.img_paths = img_paths
        self.transform = transform
        self.mask_paths = [self._get_mask_path(str(img_path)) for img_path in img_paths]

        # Verificamos que todas las máscaras existen
        for i, mask_path in enumerate(self.mask_paths):
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"No se encontró la máscara para {self.img_paths[i]}")

    @staticmethod
    def _get_mask_path(img_path: str) -> str:
        """Obtiene la ruta de la máscara correspondiente a una imagen."""
        # Obtenemos el directorio base y cambiamos 'images' por 'masks'
        img_dir = str(os.path.dirname(img_path))
        mask_dir = img_dir.replace('images', 'masks')

        # Obtenemos el nombre base del archivo (sin extensión)
        base_name = str(os.path.basename(img_path))
        base_name_without_ext = str(os.path.splitext(base_name)[0])

        # Buscamos la máscara correspondiente (puede tener extensión diferente)
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.bmp']:
            mask_path = os.path.join(str(mask_dir), f"{base_name_without_ext}_mask{ext}")
            if os.path.exists(mask_path):
                return str(mask_path)

        # Si no encontramos con ninguna extensión, devolvemos un path por defecto
        # (que generará un error controlado si no existe)
        default_mask_path = os.path.join(str(mask_dir), f"{base_name_without_ext}_mask.png")
        return str(default_mask_path)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = str(self.img_paths[idx])
        mask_path = str(self.mask_paths[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Cargar como escala de grises

        if self.transform:
            # Aplicamos la misma transformación a la imagen y la máscara
            seed = np.random.randint(2147483647)

            random.seed(seed)
            image = self.transform(image)

            random.seed(seed)
            mask = transforms.Resize(image.shape[-2:], interpolation=transforms.InterpolationMode.NEAREST)(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return image, mask


def get_transforms(train: bool = True):
    """Devuelve las transformaciones para los conjuntos de entrenamiento o validación."""
    if train:
        return transforms.Compose([
            transforms.Resize((513, 513)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((513, 513)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def prepare_dataloaders(img_dir: str, batch_size: int = 8, test_size: float = 0.15, val_size: float = 0.15):
    """Prepara los dataloaders para entrenamiento, validación y prueba."""
    # Buscamos todas las imágenes en el directorio
    img_paths = glob.glob(os.path.join(str(img_dir), "*.*"))
    # Filtramos para no incluir archivos de máscaras
    img_paths = [str(p) for p in img_paths if "_mask" not in str(p)]

    print(f"Total de imágenes encontradas: {len(img_paths)}")

    # Separamos en conjuntos de entrenamiento, validación y prueba
    train_val_paths, test_paths = train_test_split(img_paths, test_size=test_size, random_state=42)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=val_size / (1 - test_size), random_state=42)

    print(f"Imágenes de entrenamiento: {len(train_paths)}")
    print(f"Imágenes de validación: {len(val_paths)}")
    print(f"Imágenes de prueba: {len(test_paths)}")

    # Creamos los datasets
    train_dataset = OralCavityDataset(train_paths, transform=get_transforms(train=True))
    val_dataset = OralCavityDataset(val_paths, transform=get_transforms(train=False))
    test_dataset = OralCavityDataset(test_paths, transform=get_transforms(train=False))

    # Creamos los dataloaders (4, pero si estamos en Google Colab lo reducimos a 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader





