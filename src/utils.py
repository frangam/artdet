import os
import sys
import random
import math
import re
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
# from mrcnn import utils
# from mrcnn import visualize
# from mrcnn.visualize import display_images
# import mrcnn.model as modellib
# from mrcnn.model import log
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
MODEL_DIR = "model"
DAMAGE_DIR = "data"

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
def guardarImagen_ObtenerCaracteristicas(nombre_img):
    # cambios para la inferencia.
    config = damage.CustomConfig()
    # #val
    # dataset = damage.CustomDataset()
    # dataset.load_custom(DAMAGE_DIR, "val")
    # dataset.prepare()
    # print("Images value: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    #train
    dataset_train = damage.CustomDataset()
    dataset_train.load_custom(DAMAGE_DIR, "train")
    dataset_train.prepare()
    print("Images train: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        # Ejecuta la detección en una imagen a la vez
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Cambiamos el dispositivo objetivo
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    #Creamos el modelo de inferencia para las pruebas
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                config=config)
    # Cargamos el peso
    weights_path = os.path.join(MODEL_DIR,"model-maskrcnn.h5.h5")
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    carpeta_imagenes = "static/images"
    image_path =  os.path.join(carpeta_imagenes, nombre_img)
    #for image_path in image_paths:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]

    poligonos = display_instances(img, r['rois'], r['masks'], r['class_ids'],dataset_train.class_names)
    
    suma_areas = SumAreas(poligonos)

    # Calculamos las zonas que presenta el cuadro junto con el porcentaje de daño que tiene
    zonas = []
    tamaño = img.shape[0]*img.shape[1]
    zona_no_dañada = tamaño - suma_areas
    zona_dañada = tamaño - zona_no_dañada
    porcentaje = round((zona_dañada * 100)/tamaño,2)

    zonas.append(tamaño)
    zonas.append(zona_no_dañada)
    zonas.append(zona_dañada)
    zonas.append(porcentaje)

    return zonas