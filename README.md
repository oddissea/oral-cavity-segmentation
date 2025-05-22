# Segmentación de Imágenes de Cavidad Oral con DeepLabV3+

Este repositorio contiene una implementación basada en PyTorch para la segmentación semántica de imágenes de la cavidad oral, utilizando el modelo DeepLabV3+ con transfer learning.

## Descripción del proyecto

Esta implementación se ha desarrollado como parte de la PEC3 del Máster en Investigación en IA de la UNED. El objetivo principal es la segmentación de estructuras anatómicas en imágenes de la cavidad oral (dientes y lengua), utilizando técnicas de deep learning.

La implementación es compatible con diferentes plataformas de hardware (CPU, NVIDIA GPU y Apple Silicon) y está diseñada para ser fácilmente adaptable a diferentes modelos de segmentación.

## Características principales

- **Arquitectura DeepLabV3+** con backbone ResNet101 preentrenado en ImageNet
- **Transfer learning** optimizado para conjuntos de datos pequeños
- **Data augmentation** específico para imágenes médicas
- **Compatibilidad multiplataforma** (CPU, CUDA, MPS)
- **Evaluación exhaustiva** con métricas de IoU por clase y matriz de confusión
- **Visualización** de resultados de segmentación

## Estructura del proyecto

```
.
├── dataset.py        # Carga y preprocesamiento de datos
├── evaluate.py       # Funciones de evaluación y visualización
├── main.py           # Script principal
├── model.py          # Definición del modelo DeepLabV3+
├── train.py          # Funciones de entrenamiento
├── utils.py          # Funciones auxiliares
└── oral_segment_env.yml  # Entorno conda
```

## Requisitos

Para ejecutar este código, necesitarás Python 3.8+ y las siguientes bibliotecas:

- PyTorch (>=1.10)
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm
- pillow
- opencv-python

Puedes crear un entorno conda con todas las dependencias usando:

```bash
conda env create -f oral_segment_env.yml
conda activate oral_segment
```

## Uso

### Entrenamiento

```bash
python main.py --train --img_dir "ruta/dataset/images" --num_classes 3 --epochs 50
```

### Evaluación con un modelo preentrenado

```bash
python main.py --img_dir "ruta/dataset/images" --num_classes 3
```

### Inferencia en una imagen individual

```bash
python main.py --inference_image "ruta/a/imagen.jpg" --num_classes 3
```

## Resultados

El modelo consigue segmentar eficazmente las diferentes estructuras de la cavidad oral, con un mIoU (mean Intersection over Union) del X% en el conjunto de prueba.

## Licencia

Este proyecto está bajo la licencia MIT. Consulte el archivo LICENSE para más detalles.