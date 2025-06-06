from map_creation import *
from mcfunction_writer import *

def map_pipeline(image_path,
                 channels: int,
                 mirror_horizontal: bool = False,
                 world_name: str = None):
    
    image_tensor = load_image_for_map(image_path,
                                      channels = 3,
                                      mirror_horizontal = False)
    original_image, num_rows_of_maps, num_cols_of_maps = prepare_image_size_to_map(image_tensor)
    image_squares = cut_image_into_128_squares(original_image,
                                               num_rows_of_maps,
                                               num_cols_of_maps)
    for_predictions = get_img_slices_for_prediction(image_squares,
                                                    num_rows_of_maps,
                                                    num_cols_of_maps)
    maps_colors = rgb_to_ints(for_predictions,
                              image_squares,
                              num_rows_of_maps,
                              num_cols_of_maps)
    init_map, final_map, (num_rows_of_maps, num_cols_of_maps) = create_maps(maps_colors,
                                                                            num_rows_of_maps,
                                                                            num_cols_of_maps,
                                                                            world_name)
    
    return init_map, final_map, (num_rows_of_maps, num_cols_of_maps)
    
def main():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')
    print('################')
    world_name = input('world_name: ')
    if not check_world_folder_existence(world_name):
        print(f"Minecraft_World: {world_name} doesn't exists, bye bye")
        return
    image_name = input('img_name: ')
    image_path = f"images/{image_name}"
    if not os.path.exists(image_path):
        print(f'Image {image_path} does not exists, bye bye')
        return 
    
    image_extention = image_path.split('.')[1]
    out_name = f'{image_path.split('/')[1].replace('.'+image_extention,'')}'
    map_pipeline(image_path,
                 channels = 3,
                 mirror_horizontal= False,
                 world_name = world_name)

if __name__ == "__main__":
    main()