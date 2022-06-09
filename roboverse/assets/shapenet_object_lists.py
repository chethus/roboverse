TRAIN_CONTAINERS = [
    'plate',
    'cube_concave',
    'table_top',
    'bowl_small',
    'tray',
    'open_box',
    'cube',
    'torus',
]

TEST_CONTAINERS = [
    'pan_tefal',
    'marble_cube',
    'basket',
    'checkerboard_table',
]

CONTAINER_CONFIGS = {
    'plate': {
        'container_position_low': (.50, 0.22, -.30),
        'container_position_high': (.70, 0.26, -.30),
        'container_position_default': (.50, 0.22, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.46,
        'container_position_z': -0.37,
        'place_success_height_threshold': -0.32,
        'place_success_radius_threshold': 0.04,
        'min_distance_from_object': 0.11,
    },
    'cube_concave': {
        'container_name': 'cube_concave',
        'container_position_low': (.50, 0.22, -.30),
        'container_position_high': (.70, 0.26, -.30),
        'container_position_default': (.50, 0.22, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.06,
        'container_position_z': -0.35,
        'place_success_height_threshold': -0.23,
        'place_success_radius_threshold': 0.04,
        'min_distance_from_object': 0.11,
    },
    'table_top': {
        'container_position_low': (.50, 0.22, -.30),
        'container_position_high': (.70, 0.26, -.30),
        'container_position_default': (.50, 0.22, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.13,
        'container_position_z': -0.37,
        'place_success_height_threshold': -0.32,
        'place_success_radius_threshold': 0.05,
        'min_distance_from_object': 0.11,
    },
    'bowl_small': {
        'container_position_low': (.5, 0.26, -.30),
        'container_position_high': (.7, 0.26, -.30),
        'container_position_default': (.50, 0.26, -.30),
        'container_position_z': -0.35,
        'place_success_height_threshold': -0.32,
        'place_success_radius_threshold': 0.04,
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.07,
        'min_distance_from_object': 0.11,
    },
    'tray': {
        'container_position_low': (.5, 0.25, -.30),
        'container_position_high': (.7, 0.25, -.30),
        'container_position_default': (.5, 0.25, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.18,
        'container_position_z': -0.37,
        'place_success_height_threshold': -0.32,
        'place_success_radius_threshold': 0.04,
        'min_distance_from_object': 0.11,
    },
    'open_box': {
        'container_position_low': (.5, 0.23, -.30),
        'container_position_high': (.7, 0.23, -.30),
        'container_position_default': (.5, 0.23, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.1,
        'container_position_z': -0.35,
        'place_success_height_threshold': -0.32,
        'place_success_radius_threshold': 0.04,
        'min_distance_from_object': 0.11,
    },
    'pan_tefal': {
        'container_position_low': (.50, 0.22, -.30),
        'container_position_high': (.70, 0.24, -.30),
        'container_position_default': (.65, 0.23, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.4,
        'container_position_z': -0.37,
        'place_success_height_threshold': -0.32,
        'place_success_radius_threshold': 0.04,
        'min_distance_from_object': 0.1,
    },
    'husky': {
        'container_position_low': (.50, 0.22, -.30),
        'container_position_high': (.70, 0.26, -.30),
        'container_position_default': (.50, 0.22, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.13,
        'container_position_z': -0.35,
        'place_success_height_threshold': -0.23,
        'place_success_radius_threshold': 0.04,
        'min_distance_from_object': 0.10,
    },
    'marble_cube': {
        'container_position_low': (.50, 0.22, -.30),
        'container_position_high': (.70, 0.26, -.30),
        'container_position_default': (.60, 0.24, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.07,
        'container_position_z': -0.35,
        'place_success_height_threshold': -0.23,
        'place_success_radius_threshold': 0.04,
        'min_distance_from_object': 0.10,
    },
    'basket': {
        'container_name': 'basket',
        'container_position_low': (.50, 0.22, -.30),
        'container_position_high': (.70, 0.26, -.30),
        'container_position_default': (.55, 0.22, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 1.68,
        'container_position_z': -0.37,
        'place_success_height_threshold': -0.28,
        'place_success_radius_threshold': 0.04,
        'min_distance_from_object': 0.11,
    },
    'checkerboard_table': {
        'container_name': 'checkerboard_table',
        'container_position_low': (.50, 0.22, -.30),
        'container_position_high': (.70, 0.26, -.30),
        'container_position_default': (.50, 0.22, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.08,
        'container_position_z': -0.37,
        'place_success_height_threshold': -0.23,
        'place_success_radius_threshold': 0.05,
        'min_distance_from_object': 0.11,
    },
    'torus': {
        'container_position_low': (.50, 0.22, -.30),
        'container_position_high': (.70, 0.26, -.30),
        'container_position_default': (.50, 0.22, -.30),
        'container_orientation': (1, 1, 1, 1),
        'container_scale': 0.15,
        'container_position_z': -0.37,
        'place_success_height_threshold': -0.32,
        'place_success_radius_threshold': 0.04,
        'min_distance_from_object': 0.1,
    },
    'cube': {
        'container_position_low': (.5, 0.22, -.30),
        'container_position_high': (.7, 0.24, -.30),
        'container_position_default': (.5, 0.22, -.30),
        'container_orientation': (0, 0, 0.707107, 0.707107),
        'container_scale': 0.05,
        'container_position_z': -0.35,
        'place_success_radius_threshold': 0.03,
        'place_success_height_threshold': -0.23,
        'min_distance_from_object': 0.1,
    }
}

TRAIN_OBJECTS = [
    'conic_cup',
    'ball',
    'sack_vase',
    'fountain_vase',
    'shed',
    'circular_table',
    'hex_deep_bowl',
    'smushed_dumbbell',
    'square_prism_bin',
    'narrow_tray',
    # New objects:
    'colunnade_top',
    'stalagcite_chunk',
    'bongo_drum_bowl',
    'pacifier_vase',
    'beehive_funnel',
    'crooked_lid_trash_can',
    'double_l_faucet',
    'toilet_bowl',
    'pepsi_bottle',
    'two_handled_vase',
    'tongue_chair',
    'oil_tanker',
    'thick_wood_chair',
    'modern_canoe',
    'pear_ringed_vase',
    'short_handle_cup',
    'curved_handle_cup',
    'bullet_vase',
    'glass_half_gallon',
    'flat_bottom_sack_vase',
    'teepee',
    'trapezoidal_bin',
    'vintage_canoe',
    'bathtub',
    'flowery_half_donut',
    't_cup',
    'cookie_circular_lidless_tin',
    'box_sofa',
    'baseball_cap',
    'two_layered_lampshade',
]

GRASP_TRAIN_OBJECTS = [
    'conic_cup',
    'fountain_vase',
    'circular_table',
    'hex_deep_bowl',
    'smushed_dumbbell',
    'square_prism_bin',
    'narrow_tray',
    'colunnade_top',
    'stalagcite_chunk',
    'bongo_drum_bowl',
    'pacifier_vase',
    'beehive_funnel',
    'crooked_lid_trash_can',
    'toilet_bowl',
    'pepsi_bottle',
    'tongue_chair',
    'modern_canoe',
    'pear_ringed_vase',
    'short_handle_cup',
    'bullet_vase',
    'glass_half_gallon',
    'flat_bottom_sack_vase',
    'trapezoidal_bin',
    'vintage_canoe',
    'bathtub',
    'flowery_half_donut',
    't_cup',
    'cookie_circular_lidless_tin',
    'box_sofa',
    'two_layered_lampshade',
    'conic_bin',
    'jar',
    'bunsen_burner',
    'long_vase',
    'ringed_cup_oversized_base',
    'aero_cylinder',
]

PICK_PLACE_TRAIN_OBJECTS = [
    'conic_cup',
    'fountain_vase',
    'circular_table',
    'hex_deep_bowl',
    'smushed_dumbbell',
    'square_prism_bin',
    'narrow_tray',
    'colunnade_top',
    'stalagcite_chunk',
    'bongo_drum_bowl',
    'pacifier_vase',
    'beehive_funnel',
    'crooked_lid_trash_can',
    'toilet_bowl',
    'pepsi_bottle',
    'tongue_chair',
    'modern_canoe',
    'pear_ringed_vase',
    'short_handle_cup',
    'bullet_vase',
    'glass_half_gallon',
    'flat_bottom_sack_vase',
    'trapezoidal_bin',
    'vintage_canoe',
    'bathtub',
    'flowery_half_donut',
    't_cup',
    'cookie_circular_lidless_tin',
    'box_sofa',
    'two_layered_lampshade',
    'conic_bin',
    'jar',
    'aero_cylinder',
]

OBJECT_SCALINGS = {
    'conic_cup': 0.6,
    'ball': 1.0,
    'sack_vase': 0.6,
    'fountain_vase': 0.4,
    'shed': 0.6,
    'circular_table': 0.4,
    'hex_deep_bowl': 0.4,
    'smushed_dumbbell': 0.6,
    'square_prism_bin': 0.7,
    'narrow_tray': 0.35,
    # New objects:
    'colunnade_top': 0.5,
    'stalagcite_chunk': 0.6,
    'bongo_drum_bowl': 0.5,
    'pacifier_vase': 0.5,
    'beehive_funnel': 0.6,
    'crooked_lid_trash_can': 0.5,
    'double_l_faucet': 0.6,
    'toilet_bowl': 0.4,
    'pepsi_bottle': 0.65,
    'two_handled_vase': 0.45,

    'tongue_chair': 0.5,
    'oil_tanker': 1.0,
    'thick_wood_chair': 0.4,
    'modern_canoe': 0.9,
    'pear_ringed_vase': 0.65,
    'short_handle_cup': 0.5,
    'curved_handle_cup': 0.5,
    'bullet_vase': 0.6,
    'glass_half_gallon': 0.6,
    'flat_bottom_sack_vase': 0.5,

    'teepee': 0.7,
    'trapezoidal_bin': 0.4,
    'vintage_canoe': 1.0,
    'bathtub': 0.4,
    'flowery_half_donut': 0.5,
    't_cup': 0.5,
    'cookie_circular_lidless_tin': 0.5,
    'box_sofa': 0.4,
    'baseball_cap': 0.5,
    'two_layered_lampshade': 0.6,

    'conic_bin': 0.4,
    'jar': 0.8,
    'gatorade': 0.7,
    'bunsen_burner': 0.6,
    'long_vase': 0.5,
    # New objects:
    'ringed_cup_oversized_base': 0.5,
    'square_rod_embellishment': 0.6,
    'elliptical_capsule': 0.6,
    'aero_cylinder': 0.5,
    'grill_trash_can': 0.5,
}

OBJECT_BIG_SCALINGS = {
    'conic_cup': 0.9,
    'ball': 1.5,
    'sack_vase': 0.9,
    'fountain_vase': 0.6,
    'shed': 0.8,
    'circular_table': 0.6,
    'hex_deep_bowl': 0.6,
    'smushed_dumbbell': 0.9,
    'square_prism_bin': 0.9,
    'narrow_tray': 0.525,
    # New objects:
    'colunnade_top': 0.75,
    'stalagcite_chunk': 0.9,
    'bongo_drum_bowl': 0.75,
    'pacifier_vase': 0.75,
    'beehive_funnel': 0.75,
    'crooked_lid_trash_can': 0.75,
    'double_l_faucet': 0.9,
    'toilet_bowl': 0.6,
    'pepsi_bottle': 0.975,
    'two_handled_vase': 0.675,

    'tongue_chair': 0.75,
    'oil_tanker': 1.5,
    'thick_wood_chair': 0.6,
    'modern_canoe': 1.35,
    'pear_ringed_vase': 0.975,
    'short_handle_cup': 0.75,
    'curved_handle_cup': 0.75,
    'bullet_vase': 0.9,
    'glass_half_gallon': 0.9,
    'flat_bottom_sack_vase': 0.7,

    'teepee': 1.05,
    'trapezoidal_bin': 0.6,
    'vintage_canoe': 1.2,
    'bathtub': 0.6,
    'flowery_half_donut': 0.75,
    't_cup': 0.9,
    'cookie_circular_lidless_tin': 0.75,
    'box_sofa': 0.6,
    'baseball_cap': 0.6,
    'two_layered_lampshade': 0.75,

    'conic_bin': 0.6,
    'jar': 1.05,
    'gatorade': 1.05,
    'bunsen_burner': 0.9,
    'long_vase': 0.75,
    # New objects:
    'ringed_cup_oversized_base': 0.75,
    'square_rod_embellishment': 0.9,
    'elliptical_capsule': 0.8,
    'aero_cylinder': 0.7,
    'grill_trash_can': 0.75,
}

TEST_OBJECTS = [
    'conic_bin',
    'jar',
    'gatorade',
    'bunsen_burner',
    'long_vase',
    # New objects:
    'ringed_cup_oversized_base',
    'square_rod_embellishment',
    'elliptical_capsule',
    'aero_cylinder',
    'grill_trash_can',
]

GRASP_TEST_OBJECTS = [
    'square_rod_embellishment',
    'grill_trash_can',
    'shed',
    'sack_vase',
    'two_handled_vase',
    'thick_wood_chair',
    'curved_handle_cup',
    'baseball_cap',
    'elliptical_capsule',
]

PICK_PLACE_TEST_OBJECTS = [
    'square_rod_embellishment',
    'grill_trash_can',
    'shed',
    'sack_vase',
    'two_handled_vase',
    'thick_wood_chair',
    'curved_handle_cup',
    'baseball_cap',
    'elliptical_capsule',
]

ORIENTATION_OPTIONS = (
    (0, 0.707, 0.707, 0),
    (0, 0, 1, 0),
    (0, 0.707, 0, 0.707),
    (0, -0.707, 0.707, 0),
    (0.5, 0.5, 0.5, 0.5),
    (0, 0, 0.707, 0.707)
)

OBJECT_ORIENTATION_INDS = {
    'conic_cup': 1,
    'fountain_vase': 5,
    'circular_table': 2,
    'hex_deep_bowl': 0,
    'smushed_dumbbell': 1,
    'square_prism_bin': 2,
    'narrow_tray': 0,
    'colunnade_top': 5,
    'stalagcite_chunk': 5,
    'bongo_drum_bowl': 0,
    'pacifier_vase': 3,
    'beehive_funnel': 4,
    'crooked_lid_trash_can': 4,
    'toilet_bowl': 0,
    'pepsi_bottle': 5,
    'tongue_chair': 4,
    'modern_canoe': 2,
    'pear_ringed_vase': 5,
    'short_handle_cup': 4,
    'bullet_vase': 2,
    'glass_half_gallon': 2,
    'flat_bottom_sack_vase': 0,
    'trapezoidal_bin': 0,
    'vintage_canoe': 2,
    'bathtub': 3,
    'flowery_half_donut': 1,
    't_cup': 0,
    'cookie_circular_lidless_tin': 2,
    'box_sofa': 3,
    'two_layered_lampshade': 5,
    'conic_bin': 4,
    'jar': 3,
    'bunsen_burner': 2,
    'long_vase': 0,
    'ringed_cup_oversized_base': 2,
    'aero_cylinder': 5,
    'square_rod_embellishment': 5,
    'grill_trash_can': 0,
    'shed': 1,
    'sack_vase': 4,
    'two_handled_vase': 5,
    'thick_wood_chair': 1,
    'curved_handle_cup': 2,
    'baseball_cap': 2,
    'elliptical_capsule': 3,
}

OBJECT_ORIENTATIONS = {obj_name: ORIENTATION_OPTIONS[orient_ind] for obj_name, orient_ind in OBJECT_ORIENTATION_INDS.items()}

GRASP_OFFSETS = {
    'bunsen_burner': (0, 0.01, 0.0),
    'double_l_faucet': (0.01, 0.0, 0.0),
    'pear_ringed_vase': (0.0, 0.01, 0.0),
    'teepee': (0.0, 0.04, 0.0),
    'long_vase': (0.0, 0.03, 0.0),
    'ball': (0, 0, 0.0)
}
