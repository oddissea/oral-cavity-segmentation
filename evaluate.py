# Bibliotecas estándar de Python
import os

# Bibliotecas de terceros
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Tus propios módulos
from utils import process_batch


def evaluate_model(model, test_loader, criterion, device, num_classes):
    """Evalúa el modelo en el conjunto de prueba y devuelve métricas detalladas."""
    model.eval()
    test_loss = 0
    test_iou = np.zeros(num_classes)
    num_batches = len(test_loader)

    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluación") as progress:
            for images, masks in progress:
                # Utilizamos la función utils.py
                loss, _, batch_iou = process_batch(model, images, masks, criterion, device, num_classes)

                # Actualizamos contadores
                test_loss += loss
                test_iou += batch_iou

                # Actualizamos la barra de progreso
                progress.set_postfix(loss=loss, miou=batch_iou.mean())

    # Calculamos métricas promedio
    avg_test_loss = test_loss / num_batches
    avg_test_iou = test_iou / num_batches
    avg_test_miou = avg_test_iou.mean()

    # Imprimimos resultados
    print(f"Resultados en test - Loss: {avg_test_loss:.4f}, mIoU: {avg_test_miou:.4f}")
    print("\nIoU por clase:")
    for i, iou in enumerate(avg_test_iou):
        print(f"Clase {i}: IoU = {iou:.4f}")

    return avg_test_loss, avg_test_iou, avg_test_miou


def visualize_predictions(model, test_loader, device, num_samples=5, save_dir='results'):
    """Visualiza predicciones del modelo en muestras del conjunto de prueba."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    images, masks, preds = [], [], []

    # Obtenemos algunas muestras del test
    with torch.no_grad():
        for img, mask in test_loader:
            img = img.to(device)
            mask = mask.to(device)

            output = model(img)['out']
            pred = torch.argmax(output, dim=1)

            # Añadimos a nuestras listas
            for i in range(img.size(0)):
                images.append(img[i].cpu())
                masks.append(mask[i].cpu())
                preds.append(pred[i].cpu())

                if len(images) >= num_samples:
                    break

            if len(images) >= num_samples:
                break

    # Revertimos la normalización de las imágenes
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # Visualizamos los resultados
    plt.figure(figsize=(15, 5 * num_samples))

    for i in range(min(num_samples, len(images))):
        # Revertimos la normalización de la imagen
        img_show = images[i] * std + mean
        img_show = img_show.permute(1, 2, 0).numpy()
        img_show = np.clip(img_show, 0, 1)

        # Obtenemos máscara real y predicción
        mask_show = masks[i].numpy()
        pred_show = preds[i].numpy()

        # Visualizamos
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img_show)
        plt.title(f"Imagen {i + 1}")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mask_show, cmap='tab10')
        plt.title(f"Máscara real")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred_show, cmap='tab10')
        plt.title(f"Predicción")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_results.png'))
    plt.close()


def analyze_errors(model, test_loader, device, num_classes, save_dir='results'):
    """Analiza detalladamente los errores del modelo mediante una matriz de confusión."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Analizando errores"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            for cl_true in range(num_classes):
                for cl_pred in range(num_classes):
                    confusion_matrix[cl_true, cl_pred] += torch.sum((masks == cl_true) & (preds == cl_pred)).item()

    # Calculamos precision y recall por clase
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)

    for cl in range(num_classes):
        precision[cl] = confusion_matrix[cl, cl] / (confusion_matrix[:, cl].sum() + 1e-10)
        recall[cl] = confusion_matrix[cl, cl] / (confusion_matrix[cl, :].sum() + 1e-10)

    # Visualizamos la matriz de confusión
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de confusión')

    # Añadimos etiquetas
    classes = [f'Clase {i}' for i in range(num_classes)]
    plt.xticks(range(num_classes), classes, rotation=45)
    plt.yticks(range(num_classes), classes)

    # Añadimos valores
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{confusion_matrix[i, j]}",
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Imprimimos precision y recall
    print("Precision por clase:")
    for i, p in enumerate(precision):
        print(f"Clase {i}: {p:.4f}")

    print("\nRecall por clase:")
    for i, r in enumerate(recall):
        print(f"Clase {i}: {r:.4f}")

    return confusion_matrix, precision, recall


def inference_on_image(model, image_path, device, num_classes, save_dir='results'):
    """Realiza inferencia en una única imagen y visualiza el resultado.

    Args:
        :arg model: Modelo entrenado para segmentación.
        :arg image_path: Ruta a la imagen para inferencia.
        :arg device: Dispositivo donde ejecutar la inferencia (cpu, cuda, mps).
        :arg num_classes: Número de clases que el modelo puede segmentar.
        :arg save_dir: Directorio donde guardar los resultados.

    Returns:
        Predicción de segmentación como un array numpy.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Adaptamos la imagen al formato requerido por el modelo
    transform = transforms.Compose([
        transforms.Resize((513, 513)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Realizamos la predicción
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]

    # Visualizamos el resultado
    plt.figure(figsize=(14, 6))

    # Imagen original
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Imagen original")
    plt.axis('off')

    # Predicción
    plt.subplot(1, 2, 2)
    # Utilizamos num_classes para crear un mapa de colores adecuado
    cmap = plt.cm.get_cmap('tab10', num_classes)  # Crea un mapa de colores con el número exacto de clases
    plt_img = plt.imshow(pred, cmap=cmap, vmin=0, vmax=num_classes - 1)
    cbar = plt.colorbar(plt_img, ticks=range(num_classes))
    cbar.set_label('Clase')
    plt.title("Segmentación")
    plt.axis('off')

    # Guardamos la figura
    plt.tight_layout()
    output_path = os.path.join(save_dir, f'inference_{os.path.basename(image_path)}')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Resultado guardado en {output_path}")

    return pred