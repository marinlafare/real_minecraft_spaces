excluded_blocks = [
    "test_block_log",
    "crafter_north_crafting",
    "cauldron_inner",
    "grass_block_snow",
    "pointed_dripstone_down_base",
    "creaking_heart_awake",
    "jigsaw_lock",
]
# words_to_exclude = [
#     'water','front','campfire','debug','fire','lava','south','magma',
#     'top','bottom','side','plant','west','destroy','empty',
#     'north','south','east','command','inner','brewing','ice',
#     'comparator','repeater','particle','sand','conditional','chorus_flower_dead',
#     'copper_bulb','observer','test','glass','glazed',"_down_base","farmland_moist",
#     "item_frame","glow_item_frame","redstone_lamp","creaking_heart","lightning_rod",
#     "big_dripleaf_tip","chorus_flower_dead","chiseled_bookshelf_occupied","chorus_flower_dead",
#     "chiseled_bookshelf_occupied","flower","bookshelf","chiseled_nether_bricks","beehive_end","bamboo_stalk",
#     "jigsaw_lock","lectern_base"
# ]
# rename = [
#     "_down_base"
# ]


allowed_blocks = [
    # Natural & Stone-Based Blocks
    "stone",
    "granite",
    "polished_granite",
    "diorite",
    "polished_diorite",
    "andesite",
    "polished_andesite",
    "cobblestone",
    "mossy_cobblestone",
    "smooth_stone",
    "stone_bricks",
    "mossy_stone_bricks",
    "cracked_stone_bricks",
    "chiseled_stone_bricks",
    "deepslate",
    "cobbled_deepslate",
    "polished_deepslate",
    "deepslate_bricks",
    "cracked_deepslate_bricks",
    "deepslate_tiles",
    "cracked_deepslate_tiles",
    "chiseled_deepslate",
    "tuff",
    "polished_tuff",
    "tuff_bricks",
    "chiseled_tuff_bricks",
    "tuff_tiles",
    "calcite",
    "dripstone_block",
    "mud",
    "mud_bricks",
    "dirt",
    "coarse_dirt",
    "rooted_dirt",
    "podzol",
    "dirt_path",
    "sand",
    "red_sand",
    #"gravel",
    "clay",
    "obsidian",
    "crying_obsidian",
    "bedrock",
    "blackstone",
    "polished_blackstone",
    "polished_blackstone_bricks",
    "cracked_polished_blackstone_bricks",
    "chiseled_polished_blackstone",
    "gilded_blackstone",
    "basalt",
    "polished_basalt",
    "soul_sand",
    "soul_soil",
    "magma_block",
    "netherrack",
    "warped_nylium",
    "crimson_nylium",
    "nether_bricks",
    "red_nether_bricks",
    "cracked_nether_bricks",
    "quartz_block",
    "chiseled_quartz_block",
    "quartz_pillar",
    "purpur_block",
    "purpur_pillar",
    "end_stone",
    "end_stone_bricks",
    "ice",
    "packed_ice",
    "blue_ice",
    "snow_block",
    "powder_snow",
    "honey_block",
    "slime_block",

    # Wood & Plant-Based Blocks
    "oak_planks",
    "spruce_planks",
    "birch_planks",
    "jungle_planks",
    "acacia_planks",
    "dark_oak_planks",
    "crimson_planks",
    "warped_planks",
    "mangrove_planks",
    "cherry_planks",
    "bamboo_planks",
    "bamboo_mosaic",
    "oak_log",
    "spruce_log",
    "birch_log",
    "jungle_log",
    "acacia_log",
    "dark_oak_log",
    "crimson_stem",
    "warped_stem",
    "mangrove_log",
    "cherry_log",
    "bamboo_block",
    "stripped_oak_log",
    "stripped_spruce_log",
    "stripped_birch_log",
    "stripped_jungle_log",
    "stripped_acacia_log",
    "stripped_dark_oak_log",
    "stripped_crimson_stem",
    "stripped_warped_stem",
    "stripped_mangrove_log",
    "stripped_cherry_log",
    "stripped_bamboo_block",
    "oak_wood",
    "spruce_wood",
    "birch_wood",
    "jungle_wood",
    "acacia_wood",
    "dark_oak_wood",
    "crimson_hyphae", # Wood variant for nether stems
    "warped_hyphae", # Wood variant for nether stems
    "mangrove_wood",
    "cherry_wood",
    "stripped_oak_wood",
    "stripped_spruce_wood",
    "stripped_birch_wood",
    "stripped_jungle_wood",
    "stripped_acacia_wood",
    "stripped_dark_oak_wood",
    "stripped_crimson_hyphae",
    "stripped_warped_hyphae",
    "stripped_mangrove_wood",
    "stripped_cherry_wood",
    "oak_leaves",
    "spruce_leaves",
    "birch_leaves",
    "jungle_leaves",
    "acacia_leaves",
    "dark_oak_leaves",
    "mangrove_leaves",
    "cherry_leaves",
    "azalea_leaves",
    "flowering_azalea_leaves",
    "brown_mushroom_block",
    "red_mushroom_block",
    "mushroom_stem",
    "shroomlight",
    "hay_block",
    "pumpkin",
    "carved_pumpkin",
    "jack_o_lantern",
    "melon",
    "sponge",
    "wet_sponge",
    "dried_kelp_block",
    "target",
    "loom",
    "cartography_table",
    "fletching_table",
    "smithing_table",
    "barrel",
    "composter",
    "beehive",
    "bee_nest",

    # Colored & Decorative Blocks
    "white_wool", "orange_wool", "magenta_wool", "light_blue_wool", "yellow_wool", "lime_wool", "pink_wool", "gray_wool", "light_gray_wool", "cyan_wool", "purple_wool", "blue_wool", "brown_wool", "green_wool", "red_wool", "black_wool",
    "white_terracotta", "orange_terracotta", "magenta_terracotta", "light_blue_terracotta", "yellow_terracotta", "lime_terracotta", "pink_terracotta", "gray_terracotta", "light_gray_terracotta", "cyan_terracotta", "purple_terracotta", "blue_terracotta", "brown_terracotta", "green_terracotta", "red_terracotta", "black_terracotta",
    "terracotta",
    "white_concrete", "orange_concrete", "magenta_concrete", "light_blue_concrete", "yellow_concrete", "lime_concrete", "pink_concrete", "gray_concrete", "light_gray_concrete", "cyan_concrete", "purple_concrete", "blue_concrete", "brown_concrete", "green_concrete", "red_concrete", "black_concrete",
    # "white_concrete_powder", "orange_concrete_powder", "magenta_concrete_powder", "light_blue_concrete_powder", "yellow_concrete_powder", "lime_concrete_powder", "pink_concrete_powder", "gray_concrete_powder", "light_gray_concrete_powder", "cyan_concrete_powder", "purple_concrete_powder", "blue_concrete_powder", "brown_concrete_powder", "green_concrete_powder", "red_concrete_powder", "black_concrete_powder",
    "white_glazed_terracotta", "orange_glazed_terracotta", "magenta_glazed_terracotta", "light_blue_glazed_terracotta", "yellow_glazed_terracotta", "lime_glazed_terracotta", "pink_glazed_terracotta", "gray_glazed_terracotta", "light_gray_glazed_terracotta", "cyan_glazed_terracotta", "purple_glazed_terracotta", "blue_glazed_terracotta", "brown_glazed_terracotta", "green_glazed_terracotta", "red_glazed_terracotta", "black_glazed_terracotta",
    "glass",
    "white_stained_glass", "orange_stained_glass", "magenta_stained_glass", "light_blue_stained_glass", "yellow_stained_glass", "lime_stained_glass", "pink_stained_glass", "gray_stained_glass", "light_gray_stained_glass", "cyan_stained_glass", "purple_stained_glass", "blue_stained_glass", "brown_stained_glass", "green_stained_glass", "red_stained_glass", "black_stained_glass",
    "tinted_glass",
    "white_stained_glass_pane", "orange_stained_glass_pane", "magenta_stained_glass_pane", "light_blue_stained_glass_pane", "yellow_stained_glass_pane", "lime_stained_glass_pane", "pink_stained_glass_pane", "gray_stained_glass_pane", "light_gray_stained_glass_pane", "cyan_stained_glass_pane", "purple_stained_glass_pane", "blue_stained_glass_pane", "brown_stained_glass_pane", "green_stained_glass_pane", "red_stained_glass_pane", "black_stained_glass_pane",
    "glass_pane",
    "sea_lantern",
    "prismarine",
    "prismarine_bricks",
    "dark_prismarine",
    "lapis_block",
    "gold_block",
    "iron_block",
    "diamond_block",
    "emerald_block",
    "coal_block",
    "redstone_block",
    "copper_block",
    "exposed_copper",
    "weathered_copper",
    "oxidized_copper",
    "waxed_copper_block",
    "waxed_exposed_copper",
    "waxed_weathered_copper",
    "waxed_oxidized_copper",
    "cut_copper",
    "exposed_cut_copper",
    "weathered_cut_copper",
    "oxidized_cut_copper",
    "waxed_cut_copper",
    "waxed_exposed_cut_copper",
    "waxed_weathered_cut_copper",
    "waxed_oxidized_cut_copper",
    "cut_copper_stairs",
    "exposed_cut_copper_stairs",
    "weathered_cut_copper_stairs",
    "oxidized_cut_copper_stairs",
    "waxed_cut_copper_stairs",
    "waxed_exposed_cut_copper_stairs",
    "waxed_weathered_cut_copper_stairs",
    "waxed_oxidized_cut_copper_stairs",
    "cut_copper_slab",
    "exposed_cut_copper_slab",
    "weathered_cut_copper_slab",
    "oxidized_cut_copper_slab",
    "waxed_cut_copper_slab",
    "waxed_exposed_cut_copper_slab",
    "waxed_weathered_cut_copper_slab",
    "waxed_oxidized_cut_copper_slab",
    "raw_iron_block",
    "raw_copper_block",
    "raw_gold_block",
    "amethyst_block",
    "chiseled_bookshelf",
    "bookshelf",
    "note_block",
    "jukebox",
    "stonecutter",
    "furnace",
    "blast_furnace",
    "smoker",
    "crafting_table",
    "chest",
    "trapped_chest",
    "ender_chest",
    "lectern",
    "grindstone",
    "bell",
    "lantern",
    "soul_lantern",
    "end_rod",
    "glowstone",
    "redstone_lamp",
    "daylight_detector",
    "piston",
    "sticky_piston",
    "observer",
    "dropper",
    "dispenser",
    "hopper",
    "brewing_stand",
    "cauldron",
    "conduit",
    "lodestone",
    "respawn_anchor",
    "ancient_debris",
    "netherite_block",
    "chain",
    "iron_bars",
    "campfire",
    "soul_campfire",
    "enchanting_table",
    # "anvil",
    # "chipped_anvil",
    # "damaged_anvil",
    "decorated_pot",
    "sniffer_egg",
    "sculk",

    # Full Block (non-transparent) stairs and slabs (by base material) - generated with common material types
    "oak_stairs", "spruce_stairs", "birch_stairs", "jungle_stairs", "acacia_stairs", "dark_oak_stairs", "crimson_stairs", "warped_stairs", "mangrove_stairs", "cherry_stairs", "bamboo_stairs", "bamboo_mosaic_stairs",
    "stone_stairs", "cobblestone_stairs", "brick_stairs", "stone_brick_stairs", "mossy_stone_brick_stairs", "nether_brick_stairs", "red_nether_brick_stairs", "sandstone_stairs", "smooth_sandstone_stairs", "quartz_stairs", "purpur_stairs", "prismarine_stairs", "prismarine_brick_stairs", "dark_prismarine_stairs", "polished_granite_stairs", "polished_diorite_stairs", "polished_andesite_stairs", "end_stone_brick_stairs", "polished_blackstone_stairs", "polished_blackstone_brick_stairs", "cracked_polished_blackstone_brick_stairs", "deepslate_stairs", "cobbled_deepslate_stairs", "polished_deepslate_stairs", "deepslate_brick_stairs", "deepslate_tile_stairs", "mud_brick_stairs", "tuff_stairs", "polished_tuff_stairs", "tuff_brick_stairs", "tuff_tile_stairs",
    "oak_slab", "spruce_slab", "birch_slab", "jungle_slab", "acacia_slab", "dark_oak_slab", "crimson_slab", "warped_slab", "mangrove_slab", "cherry_slab", "bamboo_slab", "bamboo_mosaic_slab",
    "stone_slab", "smooth_stone_slab", "cobblestone_slab", "brick_slab", "stone_brick_slab", "mossy_stone_brick_slab", "nether_brick_slab", "red_nether_brick_slab", "sandstone_slab", "cut_sandstone_slab", "smooth_sandstone_slab", "quartz_slab", "purpur_slab", "prismarine_slab", "prismarine_brick_slab", "dark_prismarine_slab", "polished_granite_slab", "polished_diorite_slab", "polished_andesite_slab", "end_stone_brick_slab", "polished_blackstone_slab", "polished_blackstone_brick_slab", "cracked_polished_blackstone_brick_slab", "deepslate_slab", "cobbled_deepslate_slab", "polished_deepslate_slab", "deepslate_brick_slab", "deepslate_tile_slab", "mud_brick_slab", "tuff_slab", "polished_tuff_slab", "tuff_brick_slab", "tuff_tile_slab",
    # Specific missing stairs/slabs that aren't part of general patterns but are solid:
    "cut_red_sandstone_slab", "smooth_red_sandstone_slab", "red_sandstone_stairs", "smooth_red_sandstone_stairs", "cut_red_sandstone_slab", "red_sandstone_slab",
    "chiseled_polished_blackstone_stairs", # Not a "brick" variant
    "chiseled_deepslate_stairs", # Not a "brick" or "tile" variant
    "chiseled_tuff_bricks_stairs", # Not a "brick" or "tile" variant
]