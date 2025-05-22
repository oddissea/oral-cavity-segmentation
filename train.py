import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from utils import save_plot, process_batch


def train_one_epoch(model, dataloader, optimizer, criterion, device, num_classes):
    """Entrena el modelo durante una época y devuelve la pérdida y mIoU promedio."""
    model.train()
    epoch_loss = 0
    epoch_iou = np.zeros(num_classes)
    num_batches = len(dataloader)

    with tqdm(dataloader, desc="Entrenamiento") as progress:
        for images, masks in progress:
            # Utilizamos la función de utils.py
            loss, preds, batch_iou = process_batch(model, images, masks, criterion, device, num_classes)

            # Backward pass (solo en entrenamiento)
            optimizer.zero_grad()
            # Recalculamos loss para el gradiente (el devuelto por process_batch es solo el valor)
            outputs = model(images.to(device))['out']
            loss_tensor = criterion(outputs, masks.to(device))
            loss_tensor.backward()
            optimizer.step()

            # Actualizamos contadores
            epoch_loss += loss
            epoch_iou += batch_iou

            # Actualizamos la barra de progreso
            progress.set_postfix(loss=loss, miou=batch_iou.mean())

    return epoch_loss / num_batches, epoch_iou / num_batches


def validate(model, dataloader, criterion, device, num_classes):
    """Evalúa el modelo en el conjunto de validación y devuelve la pérdida y mIoU promedio."""
    model.eval()
    val_loss = 0
    val_iou = np.zeros(num_classes)
    num_batches = len(dataloader)

    with torch.no_grad():
        with tqdm(dataloader, desc="Validación") as progress:
            for images, masks in progress:
                # Utilizamos la función de utils.py
                loss, _, batch_iou = process_batch(model, images, masks, criterion, device, num_classes)

                # Actualizamos contadores
                val_loss += loss
                val_iou += batch_iou

                # Actualizamos la barra de progreso
                progress.set_postfix(loss=loss, miou=batch_iou.mean())

    return val_loss / num_batches, val_iou / num_batches


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses,
                    train_mious, val_mious, best_val_miou, save_dir, is_best=False):
    """Guarda un checkpoint completo del entrenamiento."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mious': train_mious,
        'val_mious': val_mious,
        'best_val_miou': best_val_miou,
    }

    # Guardar checkpoint regular
    checkpoint_path = os.path.join(save_dir, 'last_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)

    # Guardar mejor checkpoint si es necesario
    if is_best:
        best_checkpoint_path = os.path.join(save_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_checkpoint_path)
        # También guardamos solo el modelo (compatibilidad con código anterior)
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_deeplab_model.pth'))


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Carga un checkpoint y restaura el estado del entrenamiento."""
    if not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'train_mious': checkpoint['train_mious'],
        'val_mious': checkpoint['val_mious'],
        'best_val_miou': checkpoint['best_val_miou'],
    }


def train_model(model, train_loader, val_loader, device, num_classes, epochs=30, lr=1e-4, save_dir='models',
                resume=False):
    """Entrena el modelo con capacidad de reanudar desde checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    # Definimos criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Inicializar variables
    start_epoch = 0
    best_val_miou = 0
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []

    # Intentar cargar checkpoint si se solicita reanudar
    if resume:
        checkpoint_path = os.path.join(save_dir, 'last_checkpoint.pth')
        checkpoint_data = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

        if checkpoint_data:
            start_epoch = checkpoint_data['epoch'] + 1
            train_losses = checkpoint_data['train_losses']
            val_losses = checkpoint_data['val_losses']
            train_mious = checkpoint_data['train_mious']
            val_mious = checkpoint_data['val_mious']
            best_val_miou = checkpoint_data['best_val_miou']
            print(f"Reanudando entrenamiento desde la época {start_epoch}")
            print(f"Mejor mIoU previo: {best_val_miou:.4f}")
        else:
            print("No se encontró checkpoint. Iniciando entrenamiento desde cero.")

    # Bucle de entrenamiento
    for epoch in range(start_epoch, epochs):
        print(f"Época {epoch + 1}/{epochs}")

        # Entrenamiento
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device, num_classes)
        train_miou = train_iou.mean()
        train_losses.append(train_loss)
        train_mious.append(train_miou)

        # Validación
        val_loss, val_iou = validate(model, val_loader, criterion, device, num_classes)
        val_miou = val_iou.mean()
        val_losses.append(val_loss)
        val_mious.append(val_miou)

        # Actualizamos scheduler
        scheduler.step(val_loss)

        # Imprimimos resumen de la época
        print(f"Entrenamiento - Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}")
        print(f"Validación   - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")

        # Guardar checkpoint
        is_best = val_miou > best_val_miou
        if is_best:
            best_val_miou = val_miou
            print(f"¡Nuevo mejor modelo! mIoU: {val_miou:.4f}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            train_mious=train_mious,
            val_mious=val_mious,
            best_val_miou=best_val_miou,
            save_dir=save_dir,
            is_best=is_best
        )

        print("-" * 50)

    # Guardamos gráficas de métricas (código igual que antes)
    save_plot(train_losses, val_losses, 'Pérdida durante el entrenamiento', 'Loss', 'loss_history.png')
    save_plot(train_mious, val_mious, 'mIoU durante el entrenamiento', 'mIoU', 'miou_history.png')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mious': train_mious,
        'val_mious': val_mious,
        'best_val_miou': best_val_miou,
    }