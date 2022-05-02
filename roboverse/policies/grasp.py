import numpy as np
import roboverse.bullet as bullet

from roboverse.assets.shapenet_object_lists import GRASP_OFFSETS


class Grasp:

    def __init__(self, env, pick_height_thresh=-0.23, xyz_action_scale=7.0,
                 pick_point_noise=0.00, pick_point_z=-0.31):
        self.env = env
        self.pick_height_thresh = pick_height_thresh
        self.xyz_action_scale = xyz_action_scale
        self.pick_point_noise = pick_point_noise
        self.pick_point_z = pick_point_z
        self.reset()

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.object_to_target = self.env.object_names[
            np.random.randint(self.env.num_objects)]
        self.pick_point = bullet.get_object_position(
            self.env.objects[self.object_to_target])[0]
        if self.object_to_target in GRASP_OFFSETS.keys():
            self.pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])
        self.pick_point += np.random.normal(scale=self.pick_point_noise, size=(3,))
        self.pick_point[2] = self.pick_point_z + np.random.normal(scale=0.01)

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        done = False
        neutral_action = [0.]

        if gripper_pickpoint_dist > 0.02 and self.env.is_gripper_open:
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif self.env.is_gripper_open:
            # near the object enough, performs grasping action
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # lifting objects above the height threshold for picking
            action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # Hold
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.]

        agent_info = dict(done=done)
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info

class RotateGrasp(Grasp):
    
    def __init__(self, *args, angle_action_scale=.5, **kwargs):
        super(RotateGrasp, self).__init__(*args, **kwargs)
        self.angles_action_scale = angle_action_scale

    def reset(self):
        super().reset()
        # pitch_angle = np.random.uniform(75, 105)
        # roll_angle = np.random.uniform(-15, 15)
        pitch_angle = 90
        roll_angle = 0
        yaw_angle = np.random.uniform(90, 180) * np.random.choice((-1,1))
        self.pick_angles = np.array([pitch_angle, roll_angle, yaw_angle])
    
    def get_action(self):
        ee_pos, ee_quat = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        ee_deg = bullet.quat_to_deg(ee_quat)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        gripper_angle_dist = np.linalg.norm(self.pick_angles - ee_deg)
        done = False
        neutral_action = [0.]

        if (gripper_pickpoint_dist > 0.02 or gripper_angle_dist > 5) and self.env.is_gripper_open:
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03 or gripper_angle_dist > 7:
                action_xyz[2] = 0.0
            angle_delta = (self.pick_angles - ee_deg) % 360
            angle_delta -= (angle_delta > 180) * 360
            action_angles = angle_delta * self.angles_action_scale
            action_gripper = [0.]
        elif self.env.is_gripper_open:
            # near the object enough, performs grasping action
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # lifting objects above the height threshold for picking
            action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        else:
            # Hold
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.]

        agent_info = dict(done=done)
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info


class GraspTransfer:

    def __init__(self, env, pick_height_thresh=-0.23, xyz_action_scale=7.0,
                 pick_point_noise=0.00, suboptimal=False):
        self.env = env
        self.pick_height_thresh = pick_height_thresh
        self.xyz_action_scale = xyz_action_scale
        self.pick_point_noise = pick_point_noise
        self.suboptimal = suboptimal
        self.suboptimal_pick_point_low = (.39, .18, -.30)
        self.suboptimal_pick_point_high = (.85, .27, -.30)
        self.reset()

    def reset(self):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        self.object_to_target = self.env.object_names[
            np.random.randint(self.env.num_objects)]
        if self.suboptimal and np.random.uniform() > 0.5:
            self.pick_point = np.random.uniform(self.env.object_position_low,
                                                self.env.object_position_high)
        else:
            self.pick_point = bullet.get_object_position(
                self.env.objects[self.object_to_target])[0]
        if self.object_to_target in GRASP_OFFSETS.keys():
            self.pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])
        self.pick_point += np.random.normal(scale=self.pick_point_noise, size=(3,))
        self.pick_point[2] = -0.32 + np.random.normal(scale=0.01)

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        done = False
        neutral_action = [0.]

        if gripper_pickpoint_dist > 0.02 and self.env.is_gripper_open:
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
        elif self.env.is_gripper_open:
            # near the object enough, performs grasping action
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            # lifting objects above the height threshold for picking
            action_xyz = (self.env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [0.]
            neutral_action=[0.7]
        else:
            # Hold
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.]

        agent_info = dict(done=done)
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info


class GraspTransferSuboptimal(GraspTransfer):

    def __init__(self, env):
        super(GraspTransferSuboptimal, self).__init__(
            env, suboptimal=True)
