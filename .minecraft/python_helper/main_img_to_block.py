import os
from map_creation import check_world_folder_existence
from block_image_creation import *
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
    mcfunction_directory = assert_function_folder_exists(world_name)

    image_name = input('image name: ')
    image_path = f'images/{image_name}'
    function_name = f"blocks_{image_name}.mcfunction"
    output_path = os.path.join(mcfunction_directory, function_name)
    
    image_tensor = load_image(image_path, channels=3, mirror_horizontal= False)
    image_tensor, new_size = assert_new_proportion(image_tensor)
    blocks = get_block_from_rgb_array(image_tensor)
    write_blocks_all_facing_directions(blocks,image_name, new_size,mcfunction_directory)
    print(f'Image: {image_name} ready at: {mcfunction_directory}')

if __name__ == "__main__":
    main()