import torch
import torch.nn as nn
import argparse
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

from utils import seed_everything, get_device
from dataset import prepare_dataloaders
from model import get_model
from train import train_model
from evaluate import evaluate_model, visualize_predictions, analyze_errors, inference_on_image


def main(config):
    # Configuración de reproducibilidad
    seed_everything(config.seed)

    # Creamos directorio para este experimento concreto
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config.results_dir, f"exp_{timestamp}")
    model_dir = os.path.join(config.save_dir, f"exp_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Guardamos la configuración utilizada
    config_dict = vars(config)
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)

    # Obtención del dispositivo
    device = get_device()
    print(f"Utilizando dispositivo: {device}")

    # Preparamos los dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        img_dir=config.img_dir,
        batch_size=config.batch_size,
        test_size=config.test_size,
        val_size=config.val_size
    )

    # Creamos el modelo
    model = get_model(config.num_classes)
    model = model.to(device)

    # Entrenamos el modelo si es necesario
    if config.train:
        print(f"Iniciando entrenamiento con {config.epochs} épocas...")
        train_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_classes=config.num_classes,
            epochs=config.epochs,
            lr=config.learning_rate,
            save_dir=model_dir,
            resume=config.resume
        )

        # Guardar y visualizar las métricas de entrenamiento
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(train_metrics['train_losses']) + 1),
            'train_loss': train_metrics['train_losses'],
            'val_loss': train_metrics['val_losses'],
            'train_miou': train_metrics['train_mious'],
            'val_miou': train_metrics['val_mious']
        })

        # Guardar métricas en CSV
        metrics_path = os.path.join(experiment_dir, 'training_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Métricas de entrenamiento guardadas en {metrics_path}")

        # Generar gráficas de evolución del entrenamiento
        plt.figure(figsize=(12, 5))

        # Gráfica de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(metrics_df['epoch'], metrics_df['train_loss'], 'b-', label='Entrenamiento')
        plt.plot(metrics_df['epoch'], metrics_df['val_loss'], 'r-', label='Validación')
        plt.title('Evolución de la pérdida')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Gráfica de mIoU
        plt.subplot(1, 2, 2)
        plt.plot(metrics_df['epoch'], metrics_df['train_miou'], 'b-', label='Entrenamiento')
        plt.plot(metrics_df['epoch'], metrics_df['val_miou'], 'r-', label='Validación')
        plt.title('Evolución del mIoU')
        plt.xlabel('Época')
        plt.ylabel('mIoU')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, 'training_curves.png'))
        plt.close()

        print(f"El mejor mIoU alcanzado en validación: {train_metrics['best_val_miou']:.4f}")

    else:
        # Cargamos un modelo preentrenado
        model_path = os.path.join(config.save_dir, 'best_deeplab_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Modelo cargado desde {model_path}")
        else:
            print(f"No se encontró modelo en {model_path}. Ejecutando entrenamiento.")
            train_metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_classes=config.num_classes,
                epochs=config.epochs,
                lr=config.learning_rate,
                save_dir=model_dir
            )

            # Guardar resumen de métricas de entrenamiento en archivo de texto
            with open(os.path.join(experiment_dir, 'train_summary.txt'), 'w') as f:
                f.write(f"Entrenamiento completado con las siguientes métricas:\n")
                f.write(f"- Mejor mIoU en validación: {train_metrics['best_val_miou']:.4f}\n")
                f.write(f"- Pérdida final en entrenamiento: {train_metrics['train_losses'][-1]:.4f}\n")
                f.write(f"- Pérdida final en validación: {train_metrics['val_losses'][-1]:.4f}\n")

    # Evaluación en test
    print("\nEvaluando modelo en conjunto de prueba...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_iou, test_miou = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=config.num_classes
    )

    # Guardar resultados de evaluación
    test_results = {
        'test_loss': test_loss,
        'test_miou': test_miou,
        'test_iou_by_class': test_iou.tolist()
    }

    # Guardar en formato JSON
    with open(os.path.join(experiment_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

    # Crear tabla con IoU por clase
    class_names = ['Fondo', 'Dientes', 'Lengua']  # Ajusta según tus clases
    if len(class_names) != config.num_classes:
        class_names = [f'Clase {i}' for i in range(config.num_classes)]

    class_results_df = pd.DataFrame({
        'Clase': class_names,
        'IoU': test_iou
    })
    class_results_df.to_csv(os.path.join(experiment_dir, 'class_results.csv'), index=False)

    # Crear gráfica de barras para IoU por clase
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, test_iou)
    plt.axhline(y=test_miou, color='r', linestyle='-', label=f'mIoU: {test_miou:.4f}')
    plt.title('IoU por clase')
    plt.xlabel('Clase')
    plt.ylabel('IoU')
    plt.ylim([0, 1])
    plt.legend()

    # Añadir valores encima de las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'iou_by_class.png'))
    plt.close()

    print(f"Resultados guardados en {experiment_dir}")

    # Visualización y análisis
    print("\nGenerando visualizaciones de predicciones...")
    visualize_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
        num_samples=config.num_vis_samples,
        save_dir=experiment_dir
    )

    print("\nAnalizando errores del modelo...")
    confusion_matrix, precision, recall = analyze_errors(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=config.num_classes,
        save_dir=experiment_dir
    )

    # Guardar métricas adicionales
    metrics_df = pd.DataFrame({
        'Clase': class_names,
        'IoU': test_iou,
        'Precision': precision,
        'Recall': recall
    })
    metrics_df.to_csv(os.path.join(experiment_dir, 'detailed_metrics.csv'), index=False)

    # Crear gráfica comparativa
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25

    plt.bar(x - width, test_iou, width, label='IoU')
    plt.bar(x, precision, width, label='Precision')
    plt.bar(x + width, recall, width, label='Recall')

    plt.xlabel('Clase')
    plt.ylabel('Valor')
    plt.title('Métricas por clase')
    plt.xticks(x, class_names)
    plt.legend()
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'metrics_comparison.png'))
    plt.close()

    # Inferencia en una imagen específica si se proporciona
    if config.inference_image:
        print(f"\nRealizando inferencia en imagen: {config.inference_image}")
        pred_mask = inference_on_image(
            model=model,
            image_path=config.inference_image,
            device=device,
            num_classes=config.num_classes,
            save_dir=experiment_dir
        )

        # Opcionalmente guardar la máscara predicha como numpy array
        np.save(os.path.join(experiment_dir, 'inference_mask.npy'), pred_mask)

    print(f"\nExperimento completado. Todos los resultados disponibles en: {experiment_dir}")

    # Devolver resultados principales para posible uso posterior
    return {
        'model': model,
        'test_miou': test_miou,
        'experiment_dir': experiment_dir
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentación de imágenes de cavidad oral con DeepLabV3+")

    # Parámetros básicos
    parser.add_argument('--img_dir', type=str, default="dataset/images", help="Directorio con las imágenes")
    parser.add_argument('--num_classes', type=int, default=3, help="Número de clases (incluyendo fondo)")
    parser.add_argument('--save_dir', type=str, default="models", help="Directorio para guardar los modelos")
    parser.add_argument('--results_dir', type=str, default="results", help="Directorio para guardar resultados")

    # Parámetros de entrenamiento
    parser.add_argument('--train', action='store_true', help="Realizar entrenamiento")
    parser.add_argument('--batch_size', type=int, default=8, help="Tamaño del batch")
    parser.add_argument('--epochs', type=int, default=30, help="Número de épocas")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Tasa de aprendizaje")
    parser.add_argument('--test_size', type=float, default=0.15, help="Proporción de datos para test")
    parser.add_argument('--val_size', type=float, default=0.15, help="Proporción de datos para validación")

    # Parámetros de evaluación
    parser.add_argument('--num_vis_samples', type=int, default=5, help="Número de muestras para visualizar")
    parser.add_argument('--inference_image', type=str, default=None, help="Ruta a una imagen para realizar inferencia")

    # Otros parámetros
    parser.add_argument('--seed', type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument('--resume', action='store_true', help="Reanudar entrenamiento desde el último checkpoint")

    args = parser.parse_args()

    # Crear directorios necesarios
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Ejecutar proceso principal y capturar los resultados
    experiment_results = main(args)