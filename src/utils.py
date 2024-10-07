import os
import sys
import random
import math
import re
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util  # Para convertir SymbolicTensor a NumPy array

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
from skimage.transform import resize
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import damage
from sys import argv
import skimage

import itertools
import colorsys
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

# from Mask_RCNN.mrcnn import utils

#ROOT_DIR = os.path.abspath("C:/Users/jesus/Desktop/cosas_TFG/aplicacion/")
#ROOT_DIR = ROOT_DIR.replace('/',chr(92))
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../model/model-maskrcnn-tf")
DAMAGE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../data")

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    poligonos = []
    n_poligono = 0
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            aux = [n_poligono,p]
            poligonos.append(aux)
            ax.add_patch(p)
            
        n_poligono += 1  
    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(r"/content/static/images/deteccion.jpg", bbox_inches='tight')
    return poligonos

#----------------------------------------------------------------------------------------------
# Cálculo del área de los poligonos
#Cálculo del área de polígonos irregulares
def AreaPol(coordenadas):

  n = len(coordenadas)

  x= []
  y= []

  for i in range(n):
    x.append(float(coordenadas[i][0]))
    y.append(float(coordenadas[i][1]))

  #Algoritmo para la determinacion del area
  sum = x[0]*y[n-1] - x[n-1]*y[0]

  for i in range(n-1):
    sum += x[i+1]*y[i] - x[i]*y[i+1]
  
  area = sum/2

  return area

# Función para sumar todas las áres de los poligonos que le pasamos por parámetro
def SumAreas(poligonos):
  # Obtenemos las áreas que encontramos y la sumamos
  areas_encontradas = []
  for pl in poligonos:
    poligono_encontrado = pl[1]
    coordenadas_encontradas = poligono_encontrado.get_xy()
    area_encontrada = abs(AreaPol(coordenadas_encontradas))
    areas_encontradas.append(area_encontrada)

  suma_areas = 0
  for i in areas_encontradas:
    suma_areas += i

  return suma_areas



#------------------------------------------------------------------------------------------
#Hacemos la función
# def saveImage_GetFeatures(nombre_img):
#     # Configuración para la inferencia
#     config = damage.CustomConfig()

#     # Cargar el conjunto de datos para obtener las clases
#     dataset_train = damage.CustomDataset()
#     dataset_train.load_custom('data', "train")  # Ajusta la ruta al dataset
#     dataset_train.prepare()
#     print("Images train: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))

#     class InferenceConfig(config.__class__):
#         GPU_COUNT = 1
#         IMAGES_PER_GPU = 1
#         IMAGE_MIN_DIM = config.IMAGE_MIN_DIM
#         IMAGE_MAX_DIM = config.IMAGE_MAX_DIM

#     config = InferenceConfig()
#     config.display()

#     # Ruta de la carpeta de imágenes
#     carpeta_imagenes = "src/static/upload"
#     image_path = os.path.join(carpeta_imagenes, nombre_img)
    
#     # Leer la imagen
#     img = skimage.io.imread(image_path)

#     # Redimensionar la imagen para que coincida con las dimensiones esperadas por el modelo
#     img_resized = resize(img, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), preserve_range=True)

#     img_arr = np.array(img_resized, dtype=np.float32)
#     img_arr = np.expand_dims(img_arr, axis=0)  # Añadir dimensión extra para el lote

#     # Crear el meta-dato de la imagen (con la forma esperada por el modelo)
#     image_meta = np.array([[1024, 1024, 3] + [0] * 11])  # Ajusta según sea necesario

#     # Crear un placeholder para ROIs si no tienes (relleno de ceros)
#     rois = np.zeros((1, 1000, 4), dtype=np.float32)

#     # Cargar el modelo guardado en formato SavedModel
#     print("Loading model from SavedModel format...")
#     model = tf.saved_model.load('model/model-maskrcnn-tf')

#     # Preparar la entrada para el modelo
#     input_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)
    
#     # Realizar la inferencia utilizando la firma del modelo
#     infer = model.signatures["serving_default"]
#     results = infer(inputs=input_tensor, inputs_1=image_meta, inputs_2=rois)

#     # Mapeo de las claves disponibles a las salidas esperadas
#     print("Keys in results:", results.keys())

#     # Convertir los resultados a NumPy arrays utilizando tf.make_ndarray para extraer los datos
#     r = {
#         'rois': tensor_util.MakeNdarray(results['output_0']),      # Posibles cajas delimitadoras
#         'masks': tensor_util.MakeNdarray(results['output_3']),     # Posibles máscaras
#         'class_ids': tensor_util.MakeNdarray(results['output_1']), # Posibles IDs de clases
#         'scores': tensor_util.MakeNdarray(results['output_4'])     # Posibles puntuaciones de las detecciones
#     }

#     # Visualizar las instancias detectadas
#     poligonos = display_instances(img_resized, r['rois'], r['masks'], r['class_ids'], dataset_train.class_names)
    
#     # Calcular áreas de daño
#     suma_areas = SumAreas(poligonos)

#     # Calcular las zonas que presenta el cuadro junto con el porcentaje de daño que tiene
#     zonas = []
#     tamaño = img_resized.shape[0] * img_resized.shape[1]
#     zona_no_dañada = tamaño - suma_areas
#     zona_dañada = tamaño - zona_no_dañada
#     porcentaje = round((zona_dañada * 100) / tamaño, 2)

#     zonas.append(tamaño)
#     zonas.append(zona_no_dañada)
#     zonas.append(zona_dañada)
#     zonas.append(porcentaje)

#     return zonas







# def saveImage_GetFeatures(nombre_img):
#     # Configuración para la inferencia
#     config = damage.CustomConfig()

#     # Cargar el conjunto de datos para obtener las clases
#     dataset_train = damage.CustomDataset()
#     dataset_train.load_custom('data', "train")  # Ajusta la ruta al dataset
#     dataset_train.prepare()
#     print("Images train: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))

#     class InferenceConfig(config.__class__):
#         GPU_COUNT = 1
#         IMAGES_PER_GPU = 1
#         IMAGE_MIN_DIM = config.IMAGE_MIN_DIM
#         IMAGE_MAX_DIM = config.IMAGE_MAX_DIM

#     config = InferenceConfig()
#     config.display()

#     # Ruta de la carpeta de imágenes
#     carpeta_imagenes = "src/static/upload"
#     image_path = os.path.join(carpeta_imagenes, nombre_img)
    
#     # Leer la imagen
#     img = skimage.io.imread(image_path)

#     # Redimensionar la imagen para que coincida con las dimensiones esperadas por el modelo
#     img_resized = resize(img, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), preserve_range=True)

#     img_arr = np.array(img_resized, dtype=np.float32)
#     img_arr = np.expand_dims(img_arr, axis=0)  # Añadir dimensión extra para el lote

#     # Crear el meta-dato de la imagen (con la forma esperada por el modelo)
#     image_meta = np.array([[1024, 1024, 3] + [0] * 11])  # Ajusta según sea necesario

#     # Ajustar el tamaño de las ROIs a la cantidad que el modelo espera
#     rois = np.zeros((1, config.POST_NMS_ROIS_INFERENCE, 4), dtype=np.float32)

#     # Cargar el modelo guardado en formato SavedModel
#     print("Loading model from SavedModel format...")
#     model = tf.saved_model.load('model/model-maskrcnn-tf')

#     # Preparar la entrada para el modelo
#     input_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)

#     # Realizar la inferencia utilizando la firma del modelo
#     infer = model.signatures["serving_default"]
#     results = infer(inputs=input_tensor, inputs_1=image_meta, inputs_2=rois)

#     # To extract the tensors and evaluate them
#     rois = results['output_0']  # Posibles cajas delimitadoras
#     masks = results['output_3']  # Posibles máscaras
#     class_probabilities = results['output_1']  # Probabilidades de clase
#     scores = results['output_4']  # Puntuaciones de las detecciones

#     # If using symbolic tensors, evaluate the tensors within a session
#     if isinstance(rois, tf.Tensor):
#         with tf.compat.v1.Session() as sess:
#             rois_val = sess.run(rois)
#             masks_val = sess.run(masks)
#             class_probabilities_val = sess.run(class_probabilities)
#             scores_val = sess.run(scores)
#     else:
#         # If eager execution is on, directly use the values
#         rois_val = rois.numpy()
#         masks_val = masks.numpy()
#         class_probabilities_val = class_probabilities.numpy()
#         scores_val = scores.numpy()

#     # Obtener los IDs de clase a partir de las probabilidades
#     class_ids = np.argmax(class_probabilities_val, axis=-1)

#     # Verificar formas para detectar discrepancias
#     print(f"ROIs shape: {rois_val.shape}")
#     print(f"Masks shape: {masks_val.shape}")
#     print(f"Class IDs shape: {class_ids.shape}")

#     if rois_val.shape[0] != masks_val.shape[-1] or rois_val.shape[0] != class_ids.shape[0]:
#         raise ValueError("Mismatch between ROIs, masks, and class IDs shapes.")

#     # Visualizar las instancias detectadas
#     poligonos = display_instances(img_resized, rois_val, masks_val, class_ids, dataset_train.class_names)
    
#     # Calcular áreas de daño
#     suma_areas = SumAreas(poligonos)

#     # Calcular las zonas que presenta el cuadro junto con el porcentaje de daño que tiene
#     zonas = []
#     tamaño = img_resized.shape[0] * img_resized.shape[1]
#     zona_no_dañada = tamaño - suma_areas
#     zona_dañada = tamaño - zona_no_dañada
#     porcentaje = round((zona_dañada * 100) / tamaño, 2)

#     zonas.append(tamaño)
#     zonas.append(zona_no_dañada)
#     zonas.append(zona_dañada)
#     zonas.append(porcentaje)

#     return zonas







# def saveImage_GetFeatures(nombre_img):
#     # Configuración para la inferencia
#     config = damage.CustomConfig()

#     # Cargar el conjunto de datos para obtener las clases
#     dataset_train = damage.CustomDataset()
#     dataset_train.load_custom('data', "train")  # Ajusta la ruta al dataset
#     dataset_train.prepare()
#     print("Images train: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))

#     class InferenceConfig(config.__class__):
#         GPU_COUNT = 1
#         IMAGES_PER_GPU = 1
#         IMAGE_MIN_DIM = config.IMAGE_MIN_DIM
#         IMAGE_MAX_DIM = config.IMAGE_MAX_DIM

#     config = InferenceConfig()
#     config.display()

#     # Ruta de la carpeta de imágenes
#     carpeta_imagenes = "src/static/upload"
#     image_path = os.path.join(carpeta_imagenes, nombre_img)
    
#     # Leer la imagen
#     img = skimage.io.imread(image_path)

#     # Redimensionar la imagen para que coincida con las dimensiones esperadas por el modelo
#     img_resized = resize(img, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), preserve_range=True)

#     img_arr = np.array(img_resized, dtype=np.float32)
#     img_arr = np.expand_dims(img_arr, axis=0)  # Añadir dimensión extra para el lote

#     # Crear el meta-dato de la imagen (con la forma esperada por el modelo)
#     image_meta = np.array([[1024, 1024, 3] + [0] * 11])  # Ajusta según sea necesario

#     # Ajustar el tamaño de las ROIs a la cantidad que el modelo espera
#     rois = np.zeros((1, config.POST_NMS_ROIS_INFERENCE, 4), dtype=np.float32)

#     # Cargar el modelo guardado en formato SavedModel
#     print("Loading model from SavedModel format...")
#     model = tf.saved_model.load('model/model-maskrcnn-tf')

#     # Ensure all variables are initialized
#     print("Initializing variables...")
#     infer = model.signatures["serving_default"]

#     # Preparar la entrada para el modelo
#     input_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)

#     # Try initializing missing variables (fixing uninitialized variables)
#     try:
#         infer(inputs=input_tensor, inputs_1=image_meta, inputs_2=rois)
#     except tf.errors.FailedPreconditionError as e:
#         print(f"Warning: {e}. Attempting to initialize missing variables.")
#         # Initialize uninitialized variables
#         if hasattr(model, 'variables'):
#             for variable in model.variables:
#                 if not variable.numpy().any():
#                     print(f"Initializing variable: {variable.name}")
#                     tf.keras.backend.initialize_variables([variable])
#         # Retry inference after initializing variables
#         results = infer(inputs=input_tensor, inputs_1=image_meta, inputs_2=rois)
#     else:
#         results = infer(inputs=input_tensor, inputs_1=image_meta, inputs_2=rois)

#     # Evaluar los tensores si es necesario
#     with tf.compat.v1.Session() as sess:
#         r = {
#             'rois': sess.run(results['output_0']),
#             'masks': sess.run(results['output_3']),
#             'class_ids': sess.run(results['output_1']),
#             'scores': sess.run(results['output_4'])
#         }

#     # Visualizar las instancias detectadas
#     poligonos = display_instances(img_resized, r['rois'], r['masks'], r['class_ids'], dataset_train.class_names)
    
#     # Calcular áreas de daño
#     suma_areas = SumAreas(poligonos)

#     # Calcular las zonas que presenta el cuadro junto con el porcentaje de daño que tiene
#     zonas = []
#     tamaño = img_resized.shape[0] * img_resized.shape[1]
#     zona_no_dañada = tamaño - suma_areas
#     zona_dañada = tamaño - zona_no_dañada
#     porcentaje = round((zona_dañada * 100) / tamaño, 2)

#     zonas.append(tamaño)
#     zonas.append(zona_no_dañada)
#     zonas.append(zona_dañada)
#     zonas.append(porcentaje)

#     return zonas



def saveImage_GetFeatures(nombre_img):
    config = damage.CustomConfig()
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        # Ejecuta la detección en una imagen a la vez
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Cargar el conjunto de datos para el entrenamiento
    dataset_train = damage.CustomDataset()
    dataset_train.load_custom("data", "train")
    dataset_train.prepare()
    print("Images train: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))

    # Cargar el modelo SavedModel
    print("Cargando el modelo desde SavedModel...")
    loaded_model = tf.saved_model.load('model/model-maskrcnn-tf')
    infer = loaded_model.signatures["serving_default"]

    # Ruta de la carpeta de imágenes
    carpeta_imagenes = "src/static/upload"
    image_path = os.path.join(carpeta_imagenes, nombre_img)

    # Leer y preparar la imagen
    img = skimage.io.imread(image_path)
    if img.ndim != 3:
        # Asegurar que la imagen tenga 3 canales
        img = skimage.color.gray2rgb(img)
    img_rgb = img[..., :3].astype(np.float32)

    # Redimensionar la imagen según la configuración
    img_resized = resize(img_rgb, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), preserve_range=True)
    img_preprocessed = img_resized - config.MEAN_PIXEL  # Normalizar con el MEAN_PIXEL

    img_arr = np.expand_dims(img_preprocessed, axis=0)  # Añadir dimensión para el lote

    # Convertir a tensor
    input_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)

    # Ejecutar la inferencia
    print("Ejecutando la inferencia...")
    outputs = infer(input_tensor)

    # Procesar las predicciones
    # Inspeccionar las claves disponibles en las salidas
    print("Claves de salida del modelo:", outputs.keys())

    # Asumiendo que las salidas incluyen 'rois', 'class_ids', 'scores', 'masks'
    # Reemplaza estas claves según las salidas reales de tu SavedModel
    try:
        rois = outputs['rois'].numpy()[0]  # Shape: [N, 4]
        class_ids = outputs['class_ids'].numpy()[0]  # Shape: [N]
        scores = outputs['scores'].numpy()[0]  # Shape: [N]
        masks = outputs['masks'].numpy()  # Shape: [N, height, width, 1]
        masks = np.squeeze(masks, axis=-1)  # Shape: [N, height, width]
    except KeyError as e:
        print(f"Error: La clave {e} no se encuentra en las salidas del modelo.")
        return None

    # Visualizar las instancias detectadas
    poligonos = display_instances(img_resized, rois, masks, class_ids, dataset_train.class_names)
    
    # Calcular áreas de daño
    suma_areas = SumAreas(poligonos)

    # Calcular las zonas que presenta el cuadro junto con el porcentaje de daño que tiene
    zonas = []
    tamaño = img.shape[0] * img.shape[1]
    zona_no_dañada = tamaño - suma_areas
    zona_dañada = suma_areas  # Corregido para reflejar la suma de áreas dañadas
    porcentaje = round((zona_dañada * 100) / tamaño, 2)

    zonas.append(tamaño)
    zonas.append(zona_no_dañada)
    zonas.append(zona_dañada)
    zonas.append(porcentaje)

    return zonas