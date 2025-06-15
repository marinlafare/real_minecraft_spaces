import os
from map_creation import *
from mcfunction_writer import *
def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
def main():
    clear_screen()
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
    init_map, final_map, (num_rows_of_maps, num_cols_of_maps) = map_pipeline(image_path,
                                                                 channels = 3,
                                                                 mirror_horizontal= False,
                                                                 world_name = world_name)
    clear_screen()
    print('#########')
    print(f'word_name = {world_name}')
    print(f'img_name = {image_name}')    
    print(f'{abs(final_map-init_map)} maps created at {world_name}/data/... ')
    print('... writing mcfunctions')
    write_frame_wall(init_map,
                     final_map,
                     num_rows_of_maps,
                     num_cols_of_maps,
                     out_name,
                     world_name)
    print('... all done')
    
if __name__ == "__main__":
    main()