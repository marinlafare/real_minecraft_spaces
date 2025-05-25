import logging, os
import tensorflow as tf
from python_helper.exclude_blocks import excluded, keywords_to_exclude
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    tf.config.set_visible_devices(gpus[0], 'GPU')

def has_transparency(image_tensor):
    alpha = image_tensor[:, :, 3:4]
    for column in alpha:
        if tf.reduce_any(tf.equal(column, 0)):
            return 1
        else:
            return 0
def load_block_texture(image_path):
    try:
        image_data = tf.io.read_file(image_path)
        image = tf.io.decode_png(image_data, channels=4)
    except:
        return False
    if has_transparency(image):
        return False
    else:
        image = tf.io.decode_png(image_data, channels=3)
        return image
def get_color_mean(image_tensor):
    columns = tf.reduce_mean(image_tensor,axis = 0)
    mean_colors = tf.reduce_mean(columns,axis = 0)
    return mean_colors.numpy()
def chek_stop_words(block_name):
    for stop_word in keywords_to_exclude:
        if stop_word in block_name:
            return False
    return True
def create_color_material_dicts():
    if os.path.exists('color_dicts/color_to_material.pkl'):
        return joblib.load('color_dicts/color_to_material.pkl'),\
                joblib.load('color_dicts/material_to_color.pkl')
    else:
        available_textures = [x if x.endswith('.png') else 0 for x in os.listdir('python_helper/blocks')]
        available_textures = [x for x in available_textures if x != 0]
        
        color_to_material = dict()
        material_to_color = dict()
    
    for image in available_textures:
        #print(image)
        if image.replace('.png','') in excluded:continue
        if chek_stop_words(image):
            path = "python_helper/blocks/" + image
            image_tensor = load_block_texture(path)
        else: continue
        if type(image_tensor) == bool: continue
        else:
            mean_colors = get_color_mean(image_tensor)
            color_to_material[tuple(mean_colors)] = image
            material_to_color[image] = tuple(mean_colors)
    joblib.dump(color_to_material, 'python_helper/color_dicts/color_to_material.pkl')
    joblib.dump(material_to_color, 'python_helper/color_dicts/material_to_color.pkl')
    return color_to_material, material_to_color

def get_closest_color_model():
    color_to_material, material_to_color = create_color_material_dicts()
    #print(list(material_to_color.values()))
    available_colors = list(material_to_color.values())
    #target_color = np.array(target_color).reshape(1, -1)
    colors = np.array(available_colors)
    color_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
    color_model.fit(colors)
    return color_model, colors, color_to_material, material_to_color

color_model, colors, color_to_material, material_to_color = get_closest_color_model()

def check_proportion_result(image_path):
    pre_image = tf.io.read_file(image_path)
    pre_image = tf.io.decode_png(pre_image, channels=3)
    current_size = tuple(pre_image.shape[:-1])
    resized_image = pre_image
    correcting_proportion = True
    while correcting_proportion: 
        #print(f'X:{current_size[0]}, Y: {current_size[1]}   ')
        os.system('clear')
        is_that_ok = input(f'X:{current_size[0]}, Y: {current_size[1]}   ')
        if is_that_ok !='yes':
            new_proportion = float(input("new_proportion: "))
            pre_image = tf.io.read_file(image_path)
            pre_image = tf.io.decode_png(pre_image, channels=3)
            current_size = tuple(pre_image.shape[:-1])
            resize_size_x = int(current_size[0] * new_proportion)
            resize_size_y = int(current_size[1] * new_proportion)
            resized_image = tf.image.resize(pre_image, (resize_size_x, resize_size_y))
            resized_image = tf.cast(resized_image, dtype=tf.int32)
            current_size = tuple(resized_image.shape[:-1])
            plt.close('all')
            os.system('clear')
            plt.imshow(resized_image.numpy())
            plt.show()
            continue
        else:
            return resized_image
    
def load_image(image_path):
    pre_image = check_proportion_result(image_path)
    int_image = tf.cast(pre_image, dtype=tf.int32)
    return int_image
def setblock(x,y,z,material):
    return f"setblock ~{x} ~{y} ~{z} {material}" +"\n"
