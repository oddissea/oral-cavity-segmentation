# Segmentación Semántica de Estructuras en Cavidad Oral

Este repositorio contiene nuestro trabajo para el desarrollo de un sistema de visión artificial para la segmentación automática de estructuras anatómicas en imágenes de la cavidad oral, con el objetivo de mejorar el diagnóstico de lesiones cancerígenas.

## Contexto y motivación

El reconocimiento de cáncer oral a partir de imágenes fotográficas presenta limitaciones en su precisión diagnóstica, especialmente cuando se dispone de un número reducido de imágenes para entrenamiento. Nuestro proyecto busca mejorar estos resultados mediante la segmentación de estructuras anatómicas relevantes (dientes y lengua), proporcionando información complementaria que permita:

1. Aumentar la precisión diagnóstica del sistema de clasificación
2. Facilitar la explicabilidad del razonamiento del sistema para los profesionales clínicos
3. Optimizar el proceso diagnóstico con información estructurada y visualmente interpretable

La segmentación de estas estructuras aporta información contextual valiosa para distinguir entre tejido sano y patológico, permitiendo localizar con mayor precisión las áreas de interés diagnóstico.

## Estructura del repositorio

```
.
├── data/
│   ├── raw/                   # Imágenes originales sin procesar
│   ├── processed/             # Imágenes preprocesadas
│   ├── annotations/           # Anotaciones de segmentación
│   └── exports/               # Exportaciones y resultados
├── notebooks/
│   ├── 1_exploracion.ipynb    # Análisis exploratorio inicial
│   ├── 2_preprocesamiento.ipynb # Técnicas de preprocesamiento
│   ├── 3_sam_integration.ipynb # Integración con SAM
│   └── 4_model_training.ipynb # Entrenamiento y evaluación
├── src/
│   ├── data/                  # Funciones para gestión de datos
│   │   ├── preprocessing.py   # Preprocesamiento de imágenes
│   │   └── dataset.py         # Clase Dataset personalizada
│   ├── models/                # Implementaciones de modelos
│   │   ├── unet.py            # Implementación U-Net
│   │   └── sam_adapter.py     # Adaptador para Segment Anything
│   ├── utils/                 # Utilidades generales
│   │   ├── metrics.py         # Funciones de evaluación
│   │   └── visualization.py   # Funciones de visualización
│   └── train.py               # Script principal de entrenamiento
├── config/                    # Archivos de configuración
│   └── params.yaml            # Hiperparámetros y configuraciones
├── scripts/                   # Scripts de utilidad
│   ├── setup_environment.sh   # Configuración del entorno
│   └── label_studio_setup.py  # Configuración de Label Studio
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Este archivo
```

## Metodología

Nuestro enfoque se centra en el desarrollo de soluciones basadas en aprendizaje semi-supervisado para abordar la limitación de datos anotados, siguiendo estas etapas principales:

### 1. Preparación del entorno

Configuramos un entorno conda optimizado para hardware Apple Silicon, con soporte MPS para aceleración en PyTorch:

```bash
conda env create -f environment.yml
conda activate oral_segment
```

### 2. Obtención y procesamiento de datos

Partimos del dataset odsi-db, complementado con fuentes adicionales y datos propios anotados:

- Preprocesamiento: normalización de tamaño, ajuste de contraste y filtrado de ruido
- Aumento de datos: generación de transformaciones para diversificar el conjunto de entrenamiento
- Anotación semi-automática: uso de Segment Anything Model (SAM) integrado con Label Studio

### 3. Arquitectura del modelo

Implementamos una arquitectura U-Net adaptada con las siguientes características:

- Encoder preentrenado con ResNet50 mediante transfer learning
- Skip connections para preservar detalles espaciales
- Decoder con bloques de convolución transpuesta
- Función de pérdida combinada: Dice Loss + Binary Cross-Entropy

### 4. Flujo de trabajo de segmentación

El proceso de segmentación sigue el siguiente flujo:

1. **Preprocesamiento**: Normalización de imágenes y mejora de contraste
2. **Segmentación inicial con SAM**: Generación de máscaras preliminares
3. **Refinamiento manual**: Corrección de errores en Label Studio
4. **Entrenamiento del modelo**: Aprendizaje a partir de las anotaciones refinadas
5. **Segmentación y post-procesamiento**: Aplicación del modelo entrenado y refinamiento final

## Resultados

La evaluación de nuestro sistema produjo los siguientes resultados:

| Métrica                | Dientes | Lengua | Promedio |
|------------------------|---------|--------|----------|
| Dice Coefficient       | 0.942   | 0.879  | 0.911    |
| Jaccard Index (IoU)    | 0.891   | 0.784  | 0.838    |
| Precisión              | 0.955   | 0.914  | 0.935    |
| Exhaustividad (Recall) | 0.929   | 0.847  | 0.888    |

![Ejemplos de segmentación](https://ejemplo-url.com/images/resultados.png)

*Figura 1: Ejemplos de segmentación automática. Las regiones en azul corresponden a dientes, las regiones en verde a la lengua.*

## Instalación y uso

### Requisitos previos

- Python 3.11 o superior
- CUDA 11.7+ (para aceleración GPU) o hardware Apple Silicon (para aceleración MPS)
- 16GB+ RAM recomendados

### Configuración del entorno

```bash
# Clonar repositorio
git clone https://github.com/oddissea/oral-cavity-segmentation.git
cd oral-cavity-segmentation

# Crear y activar entorno virtual
conda env create -f environment.yml
conda activate oral_segment

# Descargar modelo SAM preentrenado
python scripts/download_sam_model.py
```

### Preparación de datos

```bash
# Estructura de directorios para datos
python scripts/prepare_directories.py

# Iniciar Label Studio para anotación
python scripts/start_label_studio.py
```

### Entrenamiento del modelo

```bash
# Ejecutar entrenamiento completo con configuración predeterminada
python src/train.py

# Ejecutar entrenamiento con parámetros personalizados
python src/train.py --config config/custom_params.yaml
```

### Inferencia

```python
from src.models import load_model
from src.utils.visualization import visualize_segmentation
import cv2

# Cargar modelo entrenado
model = load_model('path/to/model_weights.pth')

# Realizar segmentación
image = cv2.imread('path/to/image.jpg')
masks = model.predict(image)

# Visualizar resultados
visualization = visualize_segmentation(image, masks)
cv2.imwrite('result.png', visualization)
```

## Integración con Label Studio y SAM

Hemos integrado Segment Anything Model con Label Studio para facilitar el proceso de anotación:

1. Configuración inicial de Label Studio
2. Importación de imágenes de la cavidad oral
3. Aplicación de pre-anotaciones automáticas con SAM
4. Refinamiento manual de las segmentaciones
5. Exportación en formato compatible con PyTorch

Para más detalles sobre esta integración, consulta nuestro [notebook de demostración](notebooks/3_sam_integration.ipynb).

## Trabajo futuro

Identificamos las siguientes líneas de trabajo futuro:

1. Incorporación de nuevas clases de segmentación (encías, paladar, artefactos)
2. Exploración de arquitecturas Transformer para mejorar la segmentación en imágenes complejas
3. Desarrollo de un sistema end-to-end que integre segmentación y clasificación de lesiones
4. Expansión del dataset con nuevas fuentes de imágenes clínicas

## Referencias

1. Chen, L., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. In *Proceedings of the European Conference on Computer Vision (ECCV)*.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical Image Computing and Computer-Assisted Intervention* (pp. 234-241).
3. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment anything. *arXiv preprint arXiv:2304.02643*.

## Agradecimientos

- Dataset odsi-db de la Universidad del Este de Finlandia (UEF)
- Equipo de Meta AI Research por el desarrollo de SAM
- Equipo de desarrollo de Label Studio

## Autores

Fernando H. Nasser-Eddine López

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
