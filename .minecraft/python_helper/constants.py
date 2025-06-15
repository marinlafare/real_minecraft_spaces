import logging, os
import tensorflow as tf
from exclude_blocks import excluded_blocks, words_to_exclude
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
def check_stop_words(block_name):
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
    color_model = NearestNeighbors(n_neighbors=3, metric='euclidean')
    color_model.fit(colors)
    return color_model, colors, color_to_material, material_to_color

color_model, colors, color_to_material, material_to_color = get_closest_color_model()
def get_color_map_ids():
    map_colors = joblib.load('color_dicts/colors_mc_ids.pkl')
    map_color_keys = list(map_colors.keys())
    map_color_values = list(map_colors.values())
    color_to_id = dict(zip(map_color_values,map_color_keys))
    color_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(map_color_values)
    return color_model, color_to_id
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






# return f"summon minecraft:glow_item_frame ~-{x_offset} ~{y_offset} ~{z_offset} " +\
    #         "{Facing:2,Item:{id:'minecraft:filled_map',count:1,components:{'minecraft:map_id':" +\
    #         f"{n_map}"+"}}}\n"

#create_dummy_template_map(filename=output_filename)

        # Open the newly created template file for modification
        #nbt_file = None # Initialize to None for finally block
        # try:
        #     nbt_file = nbtlib.NBTFile(output_filename, "rb+") # Open in read-write binary mode
        # except FileNotFoundError:
        #     print(f"Error: Could not open newly created '{output_filename}'. Skipping this map.")
        #     n_file += 1
        #     continue
        # except Exception as e:
        #     print(f"Error opening NBT file '{output_filename}': {e}. Skipping this map.")
        #     n_file += 1
        #     continue

            


# n_file += 1 # Increment map file counter even on error
            # continue # Skip to next map if template is malformed
        # for x in range(128):
        #     for y in range(128):
        #         mc_id = color_data[count]
        #         assert type(mc_id) == np.int64
        #         nbt_file["data"]["colors"][x+(y*128)] = mc_id
        #         count+=1

 
    # import nbtlib
    # root = nbt.nbt.NBTFile()
    # root.name = title # Standard root tag name for Minecraft map files
    
    # data_compound = nbt.tag.Compound()
    # data_compound['scale'] = nbt.tag.Byte(0) # Standard map scale (0 for 1:1)
    # data_compound['dimension'] = nbt.tag.Byte(0) # Dimension ID (0 for Overworld)
    # data_compound['trackingPosition'] = nbt.tag.Byte(0) # Whether map tracks player position
    # data_compound['locked'] = nbt.tag.Byte(0) # Whether map is locked
    # data_compound['width'] = nbt.tag.Short(128) # Map width in pixels
    # data_compound['height'] = nbt.tag.Short(128) # Map height in pixels
    
    # # Initialize the 'colors' tag with a byte array of zeros (128*128 pixels).
    # # This tag will be updated with the actual image data.
    # data_compound['colors'] = nbt.tag.ByteArray([0] * (128 * 128))
    
    # root['data'] = data_compound
    
    # try:
    #     root.write_file(title)
    #     print(f"Dummy template map '{title}' created successfully.")
    # except Exception as e:
    #     print(f"Error creating dummy template map: {e}")

# def create_maps(image_path, channels = 3):
#     #image_name = image_path.split('/')[1].split('.')[0]
#     iamge_name = os.path.splitext(os.path.basename(image_path))[0]
#     maps_colors, num_rows_of_maps, num_cols_of_maps = rgb_to_ints(image_path, channels = 3)
#     n_file = get_current_map()
#     init_map = get_current_map()
    
#     print('n maps: ',len(maps_colors))
#     for ind, map_file in enumerate(maps_colors):
#         count = 0
#         nbt_file = create_template_datfile(title = f'{image_name}_map_{n_file}' )#nbt.NBTFile("example_map.dat","rb") 
#         color_data = [x[0] for x in map_file]
#         map_byte_data = bytearray(color_data)

#         try:
#             # Attempt to assign to the .value attribute (common for nbtlib.tag.ByteArray)
#             nbt_file["data"]["colors"].value = map_byte_data
#         except AttributeError:
#             # Fallback if .value doesn't exist or isn't the correct way for this library.
#             # This might replace the entire 'colors' tag object.
#             print(f"Warning: Direct '.value' assignment failed for map {n_file}. Attempting to replace 'colors' tag.")
#             # Assuming nbtlib. If you are using a different 'nbt' library,
#             # you might need to adjust `nbt.tag.ByteArray` accordingly.
#             nbt_file["data"]["colors"] = nbt.tag.ByteArray(map_byte_data)
#         except KeyError:
#             print(f"Error: 'data' or 'colors' tag not found in 'example_map.dat' for map {n_file}. Please check template structure.")
            
#         nbt_file.write_file(f'maps/map_{n_file}.dat')
#         n_file +=1
#     final_map = n_file
#     return init_map, final_map, (num_rows_of_maps, num_cols_of_maps)