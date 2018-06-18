import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import os

import sys
sys.path.append("ssd_keras")
from ssd import SSD300
from ssd_utils import BBoxUtility

'''
PIL imports
'''

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    
if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
      'nearest': pil_image.NEAREST,
      'bilinear': pil_image.BILINEAR,
      'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS
'''
'''


voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
           'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
           'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
           'Sheep', 'Sofa', 'Train', 'Tvmonitor']
CAR = 6

'''
init_model

Returns: (model, bbox_util): Parameters to be used in the main pipeline() function
'''
def init_model(weight_file='ssd_keras/SSD/weights_SSD300.hdf5'):
    
    np.set_printoptions(suppress=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    set_session(tf.Session(config=config))
    
    NUM_CLASSES = len(voc_classes) + 1
    
    input_shape=(300, 300, 3)
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    model.load_weights(weight_file, by_name=True)
    bbox_util = BBoxUtility(NUM_CLASSES)
    return model, bbox_util
   
    
'''
draw_img

Returns: canvas_img, img with the labels drawn on it
'''
def draw_img(result, img):
    
    # Parse the outputs.
    det_label = result[:, 0]
    det_conf = result[:, 1]
    det_xmin = result[:, 2]
    det_ymin = result[:, 3]
    det_xmax = result[:, 4]
    det_ymax = result[:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    canvas_img = img.copy()
    
    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        # coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        # color = colors[label]
        
        cv2.rectangle(canvas_img, (xmin, ymin), (xmax, ymax), (0,0,255), 6)
        # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        # currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
    return canvas_img


'''
from_frame

Returns: img, the PIL image version of x of size target_size
'''
def from_frame(x, target_size = None, interpolation='nearest'):
    img = pil_image.fromarray(x, 'RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError('Invalid interpolation method {} specified. Supported '
                                 'methods are {}'.format(interpolation, ', '.join(
                                     _PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img
    
    
def pipeline(frame, model_params):
    
    # Load model and bbox_util
    model, bbox_util = model_params
    
    # Read in image as data array
    inputs = []
    frame_resized = cv2.resize(frame, (300, 300)) #TODO: change this to input_shape
    img = image.img_to_array(frame_resized)
    inputs.append(img.copy())
    
    # Peprocess input
    inputs = preprocess_input(np.array(inputs))
    
    # Predict and output results from model
    preds = model.predict(inputs, batch_size=1)
    results = bbox_util.detection_out(preds)
    
    # Draw output on image
    return draw_img(results[0], frame.copy())
    
