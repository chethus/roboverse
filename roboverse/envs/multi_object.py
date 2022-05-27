from roboverse.assets.shapenet_object_lists import (
    TRAIN_OBJECTS, TRAIN_CONTAINERS, OBJECT_SCALINGS, OBJECT_BIG_SCALINGS, OBJECT_ORIENTATIONS,
    CONTAINER_CONFIGS)

import numpy as np


class MultiObjectEnv:
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """
    def __init__(self,
                 num_objects=1,
                 possible_objects=TRAIN_OBJECTS[:10],
                 use_big_scalings=False,
                 cycle_objects=False,
                 **kwargs):
        assert isinstance(possible_objects, list)
        self.cycle_objects = cycle_objects
        if cycle_objects:
            self.obj_idx = 0
        self.possible_objects = np.asarray(possible_objects)
        self.use_big_scalings = use_big_scalings
        super().__init__(**kwargs)
        self.num_objects = num_objects

    def reset(self):
        if self.cycle_objects:
            chosen_obj_idx = np.array(range(self.obj_idx, self.obj_idx + self.num_objects)) % len(self.possible_objects)
            self.obj_idx = (self.obj_idx + 1) % len(self.possible_objects)
        else:
            chosen_obj_idx = np.random.randint(0, len(self.possible_objects),
                                           size=self.num_objects)
        self.object_names = tuple(self.possible_objects[chosen_obj_idx])

        self.object_scales = dict()
        if not self.random_orientations:
            self.object_orientations = dict()
        for object_name in self.object_names:
            if not self.random_orientations:
                self.object_orientations[object_name] = OBJECT_ORIENTATIONS[object_name]
            scalings_list = OBJECT_BIG_SCALINGS if self.use_big_scalings else OBJECT_SCALINGS
            self.object_scales[object_name] = scalings_list[object_name]
        self.target_object = self.object_names[0]
        return super().reset()


class MultiObjectMultiContainerEnv:
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """
    def __init__(self,
                 num_objects=1,
                 possible_objects=TRAIN_OBJECTS[:10],
                 possible_containers=TRAIN_CONTAINERS[:3],
                 **kwargs):
        assert isinstance(possible_objects, list)
        self.possible_objects = np.asarray(possible_objects)
        self.possible_containers = np.asarray(possible_containers)

        super().__init__(**kwargs)
        self.num_objects = num_objects

    def reset(self):

        chosen_container_idx = np.random.randint(0, len(self.possible_containers))
        self.container_name = self.possible_containers[chosen_container_idx]
        container_config = CONTAINER_CONFIGS[self.container_name]
        self.container_position_low = container_config['container_position_low']
        self.container_position_high = container_config['container_position_high']
        self.container_position_z = container_config['container_position_z']
        self.container_orientation = container_config['container_orientation']
        self.container_scale = container_config['container_scale']
        self.min_distance_from_object = container_config['min_distance_from_object']
        self.place_success_height_threshold = container_config['place_success_height_threshold']
        self.place_success_radius_threshold = container_config['place_success_radius_threshold']

        chosen_obj_idx = np.random.randint(0, len(self.possible_objects),
                                           size=self.num_objects)
        self.object_names = tuple(self.possible_objects[chosen_obj_idx])
        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]
        self.target_object = self.object_names[0]
        return super().reset()
