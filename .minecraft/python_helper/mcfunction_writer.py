import os
facing_side = {'up':1,'down':6,'south': 2,'north':3,'east':4,'west':5,
              'up_z':1,'down_z':6,'south_z': 0,'north_z':2,'east_z':1,'west_z':5}
def fill_line(x1,y1,z1,x2,y2,z2,material):
    return f"fill ~{x1} ~{y1} ~{z1} ~{x2} ~{y2} ~{z2} minecraft:{material}\n"

def summon_map_line(x_offset,y_offset,z_offset,facing,n_map, rotation = 0):
    return f"summon minecraft:glow_item_frame ~-{x_offset} ~{y_offset} ~{z_offset} " \
           f"{{ItemRotation:{rotation},Facing:{facing},Item:{{id:\"minecraft:filled_map\",count:1,components:{{\"minecraft:map_id\":{n_map}}}}}}}\n"

def create_map_support(num_rows_of_maps, num_cols_of_maps, facing):
    mcfunction_lines = []
    if facing =='south':
        mcfunction_lines.append(
                        fill_line(
                            0, 
                            0, 
                            1, 
                            -(num_cols_of_maps - 1), 
                            num_rows_of_maps - 1, 
                            1,
                            'dirt'
                            )
                        )
    if facing =='north':
        mcfunction_lines.append(
                        fill_line(
                            -1, 
                            0, 
                            1, 
                            -(num_cols_of_maps)+1, 
                            num_rows_of_maps - 1, 
                            1,
                            'dirt'
                            )
                        )   
    if facing in 'west':
        mcfunction_lines.append(
                        fill_line(
                            -1, 
                            0, 
                            1, 
                            -1, 
                            num_rows_of_maps - 1, 
                            num_cols_of_maps,
                            'dirt'
                            )
                        )
    if facing in 'east':
        mcfunction_lines.append(
                        fill_line(
                            1, 
                            0, 
                            1, 
                            1, 
                            num_rows_of_maps - 1, 
                            num_cols_of_maps,
                            'dirt'
                            )
                        )
    if facing in ['up','down']:
        mcfunction_lines.append(
                        fill_line(
                            0, 
                            -1, 
                            0, 
                            -(num_rows_of_maps - 1), 
                            -1, 
                            num_cols_of_maps -1,
                            'dirt'
                            )
                        )

    return mcfunction_lines

def write_frame_wall(init_map,
                     final_map,
                     num_rows_of_maps,
                     num_cols_of_maps,
                     out_name,
                     world_name):
    current_directory = os.getcwd()
    parent = os.path.dirname(current_directory)
    mcfunction_dir = f'saves/{world_name}/datapacks/img/data/print/function/'
    for facing_direction in ['south','north','east','west','up','down']:        
        mcfunction_lines = create_map_support(num_rows_of_maps,
                                                num_cols_of_maps,
                                                facing_direction)
        for row_idx in range(num_rows_of_maps):
            for col_idx in range(num_cols_of_maps):
                current_map_id = init_map + (row_idx * num_cols_of_maps) + col_idx
                #print(facing_direction)
                if facing_direction == 'south':
                    mcfunction_lines.append(
                        summon_map_line(
                            col_idx, 
                            (num_rows_of_maps - 1 - row_idx),
                            facing_side[facing_direction+'_z'],
                            facing_side[facing_direction],
                            current_map_id))
                if facing_direction == 'north':                    
                    mcfunction_lines.append(
                        summon_map_line(
                            (num_cols_of_maps - col_idx), 
                            (num_rows_of_maps - 1 - row_idx),
                            facing_side[facing_direction+'_z'],
                            facing_side[facing_direction],
                            current_map_id))
                if facing_direction == 'east':                    
                    mcfunction_lines.append(
                        summon_map_line(
                            0, 
                            (num_rows_of_maps - 1 - row_idx),
                            col_idx+1,
                            facing_side[facing_direction],
                            current_map_id))
                if facing_direction == 'west':                    
                    mcfunction_lines.append(
                        summon_map_line(
                            0, 
                            (num_rows_of_maps - 1 - row_idx),
                            (num_cols_of_maps - 1 - col_idx)+1,
                            facing_side[facing_direction],
                            current_map_id))
                if facing_direction == 'up':                    
                    mcfunction_lines.append(
                        summon_map_line(
                            row_idx, 
                            0,
                            col_idx,
                            facing_side[facing_direction],
                            current_map_id,
                            rotation = 1
                        ))
                if facing_direction == 'down':                    
                    mcfunction_lines.append(
                        summon_map_line(
                            row_idx, 
                            -2,
                            (num_cols_of_maps - 1 - col_idx),
                            facing_side[facing_direction],
                            current_map_id,
                            rotation = 1
                        ))
            
               
        with open(os.path.join(parent,f'{mcfunction_dir}{out_name}_{facing_direction}.mcfunction'), "w") as f:
            for line in mcfunction_lines:
                f.write(line)