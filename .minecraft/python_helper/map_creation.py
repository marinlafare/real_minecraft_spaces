import os
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from itertools import chain
import tensorflow as tf
import nbtlib



def get_mcid_color_model():
    mcid_to_rgb = joblib.load('color_dicts/mcid_to_rgb.pkl')
    colors = np.array(list(mcid_to_rgb.values()))
    color_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
    color_model.fit(colors)
    return color_model, mcid_to_rgb
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
def prepare_image_size_to_map(image_tensor):
    current_height = image_tensor.shape[0]
    current_width = image_tensor.shape[1] 

    num_rows_of_maps = tf.math.ceil(current_height / 128).numpy()
    num_cols_of_maps = tf.math.ceil(current_width / 128).numpy()

    new_height = int(128 * num_rows_of_maps)
    new_width = int(128 * num_cols_of_maps)

    # resized_image = tf.image.resize(image_tensor, ((new_height, new_width)))
    # resized_image = tf.cast(resized_image, dtype=tf.int32)
    resized_image = resize(image_tensor, new_height, new_width)
    print(f'RESIZE = {num_rows_of_maps} rows, {num_cols_of_maps} columns')
    return resized_image, int(num_rows_of_maps), int(num_cols_of_maps)


def cut_image_into_128_squares(original_image, num_rows_of_maps, num_cols_of_maps):
    image_squares = []
    row_splits = tf.split(original_image, num_or_size_splits=num_rows_of_maps, axis=0)

    for row_strip in row_splits:
        col_splits = tf.split(row_strip, num_or_size_splits=num_cols_of_maps, axis=1)
        for map_square in col_splits:
            image_squares.append(map_square)

    return image_squares



def prepare_square_for_prediction(sliced_image):
    to_predict = tf.reshape(sliced_image, [128*128, 3])
    return to_predict

def get_img_slices_for_prediction(image_squares, num_rows_of_maps, num_cols_of_maps):
    for_predictions = []
    for one_square in image_squares:
        one_square_for_prediction = prepare_square_for_prediction(one_square)
        for_predictions.append(one_square_for_prediction)
    return for_predictions


def rgb_to_ints(for_predictions, image_squares, num_rows_of_maps, num_cols_of_maps):
    colors_square_ids = []
    color_model, mcid_to_rgb = get_mcid_color_model()
    for square in for_predictions:
        distances , inds = color_model.kneighbors(square)
        colors_square_ids.append(inds+4)
    return colors_square_ids


def create_template_datfile(title="some_data_idk"):
    root = nbtlib.File()
    root.name = title

    data_compound = nbtlib.Compound()
    data_compound['scale'] = nbtlib.Byte(0)
    data_compound['dimension'] = nbtlib.String("minecraft:overworld")
    data_compound['trackingPosition'] = nbtlib.Byte(0)
    data_compound['locked'] = nbtlib.Byte(1)
    data_compound['width'] = nbtlib.Short(128)
    data_compound['height'] = nbtlib.Short(128)
    
    data_compound['xCenter'] = nbtlib.Int(0)
    data_compound['zCenter'] = nbtlib.Int(0)

    #data_compound['colors'] = nbtlib.ByteArray([5] * (128 * 128))

    root['data'] = data_compound
    return root


def get_current_map(world_name,parent):
    world_data_path = os.path.join(parent, f'saves/{world_name}/data')
    map_files = os.listdir(world_data_path)
    map_files = [x for x in map_files if x.startswith('map_')]
    map_files = [x.replace('map_','').replace('.dat','') for x in map_files]
    map_files = [int(x) for x in map_files]
    if len(map_files) == 0:
        return 0
    current_map = max(map_files) + 1
    return current_map
def create_maps(maps_colors,
                num_rows_of_maps,
                num_cols_of_maps,
                world_name,
                base_image_name):
    current_directory = os.getcwd()
    parent = os.path.dirname(current_directory)
    
    n_file = get_current_map(world_name,parent)
    init_map = n_file

    print(f'Generating {len(maps_colors)} new maps.')

    for ind, map_file_data in enumerate(maps_colors):
        row = ind // num_cols_of_maps
        col = ind % num_cols_of_maps
        x_center = col * 128 + 64
        z_center = row * 128 + 64

        # output_dir = os.path.join(current_directory, "maps", base_image_name)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        
        #output_filename_local = os.path.join(output_dir, f'map_{n_file}.dat')
        world_data_path = os.path.join(parent, f'saves/{world_name}/data')

        output_filename_world = os.path.join(world_data_path, f'map_{n_file}.dat')
        nbtfile = create_template_datfile(title=f"{base_image_name}_map_{n_file}")

        color_ids = [entry[0] for entry in map_file_data]
        map_byte_data = bytearray(color_ids) 

        nbtfile["data"]["colors"] = nbtlib.ByteArray(map_byte_data)
        nbtfile["data"]["xCenter"] = nbtlib.Int(x_center)
        nbtfile["data"]["zCenter"] = nbtlib.Int(z_center)

        try:
            #nbtfile.save(output_filename_local) 
            nbtfile.save(output_filename_world) 
        except Exception as e:
            print(f"Error saving map_{n_file}.dat: {e}")

        n_file += 1

    final_map = n_file
    print(f"Map generation complete. Generated maps from ID {init_map} to {final_map - 1}.")
    return init_map, final_map, (num_rows_of_maps, num_cols_of_maps)
def check_world_folder_existence(world_name):
    current_directory = os.getcwd()
    parent = os.path.dirname(current_directory)
    world_directory = os.path.join(parent, 'saves',world_name,'data')
    return os.path.exists(world_directory)

def map_pipeline(image_path, channels: int, mirror_horizontal: bool = False, world_name: str = None):
    if not check_world_folder_existence(world_name):
        return f"Minecraft_World: {world_name} doesn't exists"
    base_image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_tensor = load_image(image_path, channels = 3, mirror_horizontal = False)
    original_image, num_rows_of_maps, num_cols_of_maps = prepare_image_size_to_map(image_tensor)
    image_squares = cut_image_into_128_squares(original_image, num_rows_of_maps, num_cols_of_maps)
    for_predictions = get_img_slices_for_prediction(image_squares, num_rows_of_maps, num_cols_of_maps)
    maps_colors = rgb_to_ints(for_predictions, image_squares, num_rows_of_maps, num_cols_of_maps)
    init_map, final_map, (num_rows_of_maps, num_cols_of_maps) = create_maps(maps_colors,
                                                                            num_rows_of_maps,
                                                                            num_cols_of_maps,
                                                                            world_name,
                                                                            base_image_name = base_image_name)
    
    return init_map, final_map, (num_rows_of_maps, num_cols_of_maps)




# import os
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# from itertools import chain
# import tensorflow as tf
# import nbtlib





# def load_image_for_map(image_path, channels: int, mirror_horizontal: bool = False):
#     image_bytes = tf.io.read_file(image_path)
#     image_tensor = tf.image.decode_image(image_bytes, channels=channels) 

#     if mirror_horizontal:
#         image_tensor = tf.image.flip_left_right(image_tensor)

#     int_image = tf.cast(image_tensor, dtype=tf.int32)
#     return int_image

# def prepare_image_size_to_map(image_path, channels: int):
#     image_tensor = load_image_for_map(image_path, channels = channels)
#     current_height = image_tensor.shape[0]
#     current_width = image_tensor.shape[1] 

#     num_rows_of_maps = tf.math.ceil(current_height / 128).numpy()
#     num_cols_of_maps = tf.math.ceil(current_width / 128).numpy()

#     new_height = int(128 * num_rows_of_maps)
#     new_width = int(128 * num_cols_of_maps)

#     resized_image = tf.image.resize(image_tensor, ((new_height, new_width)))
#     resized_image = tf.cast(resized_image, dtype=tf.int32)
#     print(f'RESIZE = {num_rows_of_maps} rows, {num_cols_of_maps} columns')
#     return resized_image, int(num_rows_of_maps), int(num_cols_of_maps)

# def cut_image_into_128_squares(image_path, channels: int):
#     image_squares = []
#     original_image, num_rows_of_maps, num_cols_of_maps = prepare_image_size_to_map(image_path, channels)
#     row_splits = tf.split(original_image, num_or_size_splits=num_rows_of_maps, axis=0)

#     for row_strip in row_splits:
#         col_splits = tf.split(row_strip, num_or_size_splits=num_cols_of_maps, axis=1)
#         for map_square in col_splits:
#             image_squares.append(map_square)

#     return image_squares, num_rows_of_maps, num_cols_of_maps
 
# def prepare_square_for_prediction(sliced_image):
#     to_predict = tf.reshape(sliced_image, [128*128, 3])
#     return to_predict

# def get_img_slices_for_prediction(image_path, channels = 3):
#     for_predictions = [] 
#     image_squares, num_rows_of_maps, num_cols_of_maps = cut_image_into_128_squares(image_path, channels = 3)
#     for one_square in image_squares:
#         one_square_for_prediction = prepare_square_for_prediction(one_square)
#         for_predictions.append(one_square_for_prediction)
#     return for_predictions, image_squares, num_rows_of_maps, num_cols_of_maps

# def rgb_to_ints(image_path, channels = 3):
#     for_predictions, image_squares, num_rows_of_maps, num_cols_of_maps = get_img_slices_for_prediction(image_path, channels = 3)
#     colors_square_ids = []
#     color_model, mcid_to_rgb = get_mcid_color_model()
#     for square in for_predictions:
#         distances , inds = color_model.kneighbors(square)
#         colors_square_ids.append(inds+4) 
#     return colors_square_ids, num_rows_of_maps, num_cols_of_maps

# def create_template_datfile(title="some_data_idk"):
#     root = nbtlib.File()
#     root.name = title

#     data_compound = nbtlib.Compound()
#     data_compound['scale'] = nbtlib.Byte(0)
#     data_compound['dimension'] = nbtlib.String("minecraft:overworld")
#     data_compound['trackingPosition'] = nbtlib.Byte(0)
#     data_compound['locked'] = nbtlib.Byte(1)
#     data_compound['width'] = nbtlib.Short(128)
#     data_compound['height'] = nbtlib.Short(128)
    
#     data_compound['xCenter'] = nbtlib.Int(0)
#     data_compound['zCenter'] = nbtlib.Int(0)

#     data_compound['colors'] = nbtlib.ByteArray([5] * (128 * 128))

#     root['data'] = data_compound
#     return root

# def create_maps(image_path, world_name, channels=3):
#     current_directory = os.getcwd()
#     parent = os.path.dirname(current_directory)
#     base_image_name = os.path.splitext(os.path.basename(image_path))[0]
#     maps_colors, num_rows_of_maps, num_cols_of_maps = rgb_to_ints(image_path, channels=channels)
#     n_file = get_current_map()
#     init_map = n_file

#     print(f'Generating {len(maps_colors)} new maps.')

#     for ind, map_file_data in enumerate(maps_colors):
#         row = ind // num_cols_of_maps
#         col = ind % num_cols_of_maps
#         x_center = col * 128 + 64
#         z_center = row * 128 + 64

#         output_dir = os.path.join(current_directory, "maps", base_image_name)
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
        
#         output_filename_local = os.path.join(output_dir, f'map_{n_file}.dat')
#         world_data_path = os.path.join(parent, f'saves/{world_name}/data')

#         output_filename_world = os.path.join(world_data_path, f'map_{n_file}.dat')
#         nbtfile = create_template_datfile(title=f"{base_image_name}_map_{n_file}")

#         color_ids = [entry[0] for entry in map_file_data]
#         map_byte_data = bytearray(color_ids) 

#         nbtfile["data"]["colors"] = nbtlib.ByteArray(map_byte_data)
#         nbtfile["data"]["xCenter"] = nbtlib.Int(x_center)
#         nbtfile["data"]["zCenter"] = nbtlib.Int(z_center)

#         try:
#             nbtfile.save(output_filename_local) 
#             nbtfile.save(output_filename_world) 
#         except Exception as e:
#             print(f"Error saving map_{n_file}.dat: {e}")

#         n_file += 1

#     final_map = n_file
#     print(f"Map generation complete. Generated maps from ID {init_map} to {final_map - 1}.")
#     return init_map, final_map, (num_rows_of_maps, num_cols_of_maps)

    
    
# def get_current_map():
#     current_map_files = [x for x in os.listdir('maps') if x.endswith('.dat')]
#     available_map_ids = [int(x.replace('map_','').replace('.dat','')) for x in current_map_files]
#     if len(available_map_ids) == 0:
#         return 0
#     else:
#         return max(available_map_ids) + 1

# def fill_line(x1,y1,z1,x2,y2,z2,material):
#     return f"fill ~{x1} ~{y1} ~{z1} ~{x2} ~{y2} ~{z2} minecraft:{material}\n"

# def summon_map_line(x_offset,y_offset,z_offset,facing,n_map, rotation = 0):
#     return f"summon minecraft:glow_item_frame ~-{x_offset} ~{y_offset} ~{z_offset} " \
#            f"{{ItemRotation:{rotation},Facing:{facing},Item:{{id:\"minecraft:filled_map\",count:1,components:{{\"minecraft:map_id\":{n_map}}}}}}}\n"
# def create_map_support(num_rows_of_maps, num_cols_of_maps, map_range,facing):
#     mcfunction_lines = []
#     if facing =='south':
#         mcfunction_lines.append(
#                         fill_line(
#                             0, 
#                             0, 
#                             1, 
#                             -(num_cols_of_maps - 1), 
#                             num_rows_of_maps - 1, 
#                             1,
#                             'dirt'
#                             )
#                         )
#     if facing =='north':
#         mcfunction_lines.append(
#                         fill_line(
#                             -1, 
#                             0, 
#                             1, 
#                             -(num_cols_of_maps)+1, 
#                             num_rows_of_maps - 1, 
#                             1,
#                             'dirt'
#                             )
#                         )   
#     if facing in 'west':
#         mcfunction_lines.append(
#                         fill_line(
#                             -1, 
#                             0, 
#                             1, 
#                             -1, 
#                             num_rows_of_maps - 1, 
#                             num_cols_of_maps,
#                             'dirt'
#                             )
#                         )
#     if facing in 'east':
#         mcfunction_lines.append(
#                         fill_line(
#                             1, 
#                             0, 
#                             1, 
#                             1, 
#                             num_rows_of_maps - 1, 
#                             num_cols_of_maps,
#                             'dirt'
#                             )
#                         )
#     if facing in ['up','down']:
#         mcfunction_lines.append(
#                         fill_line(
#                             0, 
#                             -1, 
#                             0, 
#                             -(num_rows_of_maps - 1), 
#                             -1, 
#                             num_cols_of_maps -1,
#                             'dirt'
#                             )
#                         )
#     first_map = map_range[0]
#     last_map = map_range[1]
#     maps_numbers = [int(x.replace('map_','').replace('.dat','')) for x in os.listdir('maps') if x.endswith('.dat')]
#     maps_numbers = [x for x in maps_numbers if x in range(first_map,last_map)]
#     maps_numbers = sorted(maps_numbers)

#     return mcfunction_lines, maps_numbers
    
# def write_frame_wall(num_rows_of_maps, num_cols_of_maps, map_range, out_name,world_name):
#     current_directory = os.getcwd()
#     parent = os.path.dirname(current_directory)
#     mcfunction_dir = f'saves/{world_name}/datapacks/img/data/print/function/'
    
#     for facing_direction in ['south','north','east','west','up','down']:
        
#         mcfunction_lines, maps_numbers = create_map_support(num_rows_of_maps,
#                                                                         num_cols_of_maps,
#                                                                         map_range, facing_direction)
#         for row_idx in range(num_rows_of_maps):
#             for col_idx in range(num_cols_of_maps):
#                 current_map_id = map_range[0] + (row_idx * num_cols_of_maps) + col_idx
#                 if facing_direction == 'south':
#                     mcfunction_lines.append(
#                         summon_map_line(
#                             col_idx, 
#                             (num_rows_of_maps - 1 - row_idx),
#                             facing_side[facing_direction+'_z'],
#                             facing_side[facing_direction],
#                             current_map_id))
#                 if facing_direction == 'north':
#                     mcfunction_lines.append(
#                         summon_map_line(
#                             (num_cols_of_maps - col_idx), 
#                             (num_rows_of_maps - 1 - row_idx),
#                             facing_side[facing_direction+'_z'],
#                             facing_side[facing_direction],
#                             current_map_id))
#                 if facing_direction == 'east':
#                     mcfunction_lines.append(
#                         summon_map_line(
#                             0, 
#                             (num_rows_of_maps - 1 - row_idx),
#                             col_idx+1,
#                             facing_side[facing_direction],
#                             current_map_id))
#                 if facing_direction == 'west':
#                     mcfunction_lines.append(
#                         summon_map_line(
#                             0, 
#                             (num_rows_of_maps - 1 - row_idx),
#                             (num_cols_of_maps - 1 - col_idx)+1,
#                             facing_side[facing_direction],
#                             current_map_id))
#                 if facing_direction == 'up':
#                     mcfunction_lines.append(
#                         summon_map_line(
#                             row_idx, 
#                             0,
#                             col_idx,
#                             facing_side[facing_direction],
#                             current_map_id,
#                             rotation = 1
#                         ))
#                 if facing_direction == 'down':
#                     mcfunction_lines.append(
#                         summon_map_line(
#                             row_idx, 
#                             -2,
#                             (num_cols_of_maps - 1 - col_idx),
#                             facing_side[facing_direction],
#                             current_map_id,
#                             rotation = 1
#                         ))
            
               
#             with open(os.path.join(parent,
#                 f'{mcfunction_dir}{out_name}_{facing_direction}.mcfunction'), "w") as f:
#                 for line in mcfunction_lines:
#                     f.write(line)
#         else: pass
        

 