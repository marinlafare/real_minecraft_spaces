import os
from IPython.display import clear_output
from exclude_blocks import *
import joblib

from map_creation import check_world_folder_existence
import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

material_to_color = joblib.load('color_dicts/material_to_color.pkl')

def setblock(x,y,z,material):
    return f"setblock ~{x} ~{y} ~{z} {material}" +"\n"
def assert_function_folder_exists(world_name):
    current_directory = os.getcwd()
    parent = os.path.dirname(current_directory)
    mcfunction_directory = os.path.join(parent,f'saves/{world_name}/datapacks/img/data/print/function')
    if os.path.exists(mcfunction_directory):
        return mcfunction_directory
    else:
        return False
def load_image(image_path, channels: int, mirror_horizontal: bool = False):
    image_bytes = tf.io.read_file(image_path)
    image_tensor = tf.image.decode_image(image_bytes, channels=channels) 
    if mirror_horizontal:
        image_tensor = tf.image.flip_left_right(image_tensor)
    int_image = tf.cast(image_tensor, dtype=tf.int32)
    return int_image
def resize(image_tensor, new_height, new_width):
    resized_image = tf.image.resize(image_tensor, ((new_height, new_width)))
    resized_image = tf.cast(resized_image, dtype=tf.int32)
    return resized_image
def assert_new_proportion(image_tensor):
    current_size = image_tensor.shape[:2]
    new_size = False
    while True:
        if not new_size:
            is_this_ok = input(f'Is this ok: X:{current_size[0]}, Y: {current_size[1]}   ')
        else:
            is_this_ok = input(f'Is this ok: X:{new_size[0]}, Y: {new_size[1]}   ')
        if is_this_ok != 'yes':
            new_proportion = input('ok, new scale then: ')
            try:
                new_proportion = float(new_proportion)
            except:
                print(new_proportion, 'must be a float number, eg 0.2')
                continue
            new_size = [np.ceil(side * float(new_proportion)) for side in current_size]
            print(new_size)
        else:
            if new_size:
                image_tensor = resize(image_tensor, int(new_size[0]), int(new_size[1]))
                return image_tensor, [int(x) for x in new_size]
            else:
                image_tensor = resize(image_tensor, current_size[0], current_size[1])
                return image_tensor, [int(x) for x in current_size]



def filter_block_from_model(material_to_color):
    available_materials = list(material_to_color.keys())
    available_materials = [x.replace('.png','') for x in material_to_color.keys()]
    
    filtered_material_to_color = dict()
    for material in available_materials:
        if material in allowed_blocks:
            filtered_material_to_color[material] = material_to_color[material+'.png']
    filtered_color_to_material = dict(zip(filtered_material_to_color.values(),
                                         filtered_material_to_color.keys()))
    return filtered_material_to_color, filtered_color_to_material
            
def get_image_to_blocks_color_model(filtered_material_to_color):
    available_colors = list(filtered_material_to_color.values())
    colors = np.array(available_colors)
    color_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
    color_model.fit(colors)
    return color_model, colors
def get_block_from_rgb_array(image_tensor):
    import random
    x = image_tensor.shape[0]
    y = image_tensor.shape[1]
    
    filtered_material_to_color, filtered_color_to_material = filter_block_from_model(material_to_color)    
    color_model, colors = get_image_to_blocks_color_model(filtered_material_to_color)
    for_prediction = tf.reshape(image_tensor, [x*y, 3])
    # probabilities = [0.75, 0.15, 0.1]
    dist, prediction = color_model.kneighbors(for_prediction)
    prediction = [colors[x] for x in prediction]
    blocks = [filtered_color_to_material[tuple(prediction[x][0])] for x,y in enumerate(prediction)]
    
    return blocks

def write_blocks_all_facing_directions(blocks,image_name, new_size, mcfunction_directory):
    
    for facing in ['floor', 'wall']:
        
        output = f"{mcfunction_directory}/blocks_{image_name.split('.')[0]}_{facing}_{new_size[0]}_{new_size[1]}.mcfunction"
        current_block = 0
        mcfunction_lines = []
        for columns in range(new_size[0]):
            for rows in range(new_size[1]):
                if facing == 'floor':
                    mcfunction_lines.append(setblock(columns,0,rows,blocks[current_block]))
                else:
                    mcfunction_lines.append(setblock(columns,rows,0,blocks[current_block]))
                current_block +=1
        with open(output, "w") as f:
            for line in mcfunction_lines:
                f.write(line)