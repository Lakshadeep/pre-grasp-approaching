# Isaac imports

from omni.isaac.kit import SimulationApp


# Enables livestream server to connect to when running headless
# CONFIG = {
#     "width": 1920, #1280,
#     "height": 1080, #720,
#     "window_width": 2120,
#     "window_height": 1280,
#     "headless": False,
#     "renderer": "RayTracedLighting",
#     # "display_options": 3286,  # Set display options to show default grid,
# }

# Disable livestream server to connect to when running headless
CONFIG = {
    "width": 640,
    "height": 480,
    "window_width": 640,
    "window_height": 480,
    "headless": True,
    "renderer": "RayTracedLighting",
    # "display_options": 3286,  # Set display options to show default grid,
}

# Start the omniverse application
simulation_app = SimulationApp(launch_config=CONFIG)

# Default Livestream settings
simulation_app.set_setting("/app/window/drawMouse", True)
simulation_app.set_setting("/app/livestream/proto", "ws")
simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 10)
simulation_app.set_setting("/ngx/enabled", False)


import omni.kit
from utils.synthetic_data import SyntheticDataHelper
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.dynamic_control import _dynamic_control
import omni.graph.core as og
from omni.isaac.core_nodes.scripts.utils import set_target_prims    
from omni.isaac.core.utils.extensions import enable_extension
from pxr import UsdLux, UsdGeom, Sdf, Gf, UsdPhysics

from omni.isaac.core.robots import Robot
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper

from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation.lula import LulaTaskSpaceTrajectoryGenerator # not available in Isaac 1.1
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation import ArticulationTrajectory # not available in Isaac 1.1

# enable websocket extension
enable_extension("omni.services.streamclient.websocket")

# general python libraries
import gym
from gym import spaces
import torch
import math
import numpy as np
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation
import datetime
from datetime import time
import cv2
import time
import random

# Mushroom rl imports
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces
from mushroom_rl.utils.angles import normalize_angle
from mushroom_rl.utils.viewer import Viewer


class Task(Environment):
    """
    Simple room environment for single object with ER mobile manipulator and onrobotic 2f gripper
    """

    ur5e_default_joint_angles = [-180, -55.5, 56, -175, 268, 0]

    def __init__(
        self ,
        cfg
    ) -> None:
        self.cfg = cfg
        gamma = cfg.mdp.gamma                                
        horizon = cfg.mdp.horizon

        observation_space = spaces.Box(low=np.array([-3,-3,-3,-1,-1,-1,-1]), 
                                        high=np.array([3,3,3,1,1,1,1]))

        self._max_dist = cfg.mdp.max_dist
        self._max_base_turn = cfg.mdp.max_base_turn

        # we dont support backward motion
        action_space = spaces.Box(low=np.array([0, -self._max_base_turn]),
                                  high=np.array([self._max_dist, self._max_base_turn]))
        
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        self.initialize_mdp(mdp_info)
        self.initialize_parameters()

        # initializes the episode scene
        self._set_up_scene()


    def initialize_mdp(self, mdp_info):
        super().__init__(mdp_info)


    def initialize_parameters(self):
        # task-specific parameters
        self._robot_base_pose = [0.0, 0.0, 2.0] # x, y, theta

        # Initialize Isaac API for getting synthetic data
        self.synthetic_data = SyntheticDataHelper()
        self.viewport_interface = omni.kit.viewport_legacy.get_viewport_interface()

        # some private variables
        self._state = np.zeros((7,))
        self._random = True
        self._n_steps = 0

        # UR5e default joint angles in degrees
        self.ur5e_default_joint_angles = [-180, -55.5, 56, -175, 268, 0]

        # UR5e joint limits in degrees
        # lower bound
        self.ur5e_joint_lower_limits = [-2*np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]

        # upper boound
        self.ur5e_joint_upper_limits = [2*np.pi, 0, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]

        self.curr_ik_solution = np.zeros((6,))


    def reset(self, state=None, robot_r_min=1.5, robot_r_max=3.0, table_r_max=0.4):
        self._n_steps = 0
        self.reset_arm()

        if state is None:
            if self._random:
                # we only randomize angle only around z axis
                angle_z = np.random.uniform(-np.pi, np.pi)
            else:
                angle_z = np.pi

            robot_r = np.random.uniform(robot_r_min, robot_r_max)
            theta = np.random.uniform(-np.pi, np.pi)

            robot_x = robot_r * np.cos(theta)
            robot_y = robot_r * np.sin(theta)

            # This should ensure that object stays in the field of view when we reset
            robot_theta = self._wrap_angle(theta + np.random.uniform(-np.pi/9, np.pi/9) + np.pi)
            self.move_robot_base_to_pose(robot_x, robot_y, robot_theta)

            self.get_grasp_poses(self.object_prim.get_world_pose(), self.ur5e_prim.get_world_pose())
        else:
            # TODO
            pass

        self._state = self._get_state()
        return self._state

    
    def shutodwn(self):
        simulation_app.close()

    
    def add_noise(self, pose, t_sigma, r_sigma):
        pose[0][0] = pose[0][0] +  (t_sigma * random.uniform(-0.01, 0.01))
        pose[0][1] = pose[0][1] +  (t_sigma * random.uniform(-0.01, 0.01))
        pose[0][2] = pose[0][2] +  (t_sigma * random.uniform(-0.01, 0.01))

        robot_rot_rpy = Rotation.from_quat([pose[1][1], pose[1][2], pose[1][3], pose[1][0]]).as_euler('xyz')
        robot_rot_rpy[2] = self._wrap_angle(robot_rot_rpy[2] + (r_sigma * random.uniform(-0.01, 0.01)))
        quat = Rotation.from_euler('xyz', robot_rot_rpy).as_quat()

        pose[1][1] = quat[0]
        pose[1][2] = quat[1]
        pose[1][3] = quat[2]
        pose[1][0] = quat[3]
        return pose


    def _get_state(self):
        obj_pose = self.object_prim.get_world_pose()
        robot_pose = self.robot_base_prim.get_world_pose()
        dist = np.linalg.norm(obj_pose[0][0:2] - robot_pose[0][0:2])

        # obj_pose = self.add_noise(obj_pose, dist, dist)

        wTo = self._isaac_pose_to_transformation_matrix(obj_pose)
        # wTc = self._isaac_pose_to_transformation_matrix(self.camera_prim.get_world_pose())  # camera frame
        wTr = self._isaac_pose_to_transformation_matrix(self.ur5e_prim.get_world_pose())  # robot frame


        # for learning in robot frame
        rTw = np.linalg.inv(wTr)
        rTo = np.matmul(rTw, wTo)
        quat = Rotation.from_matrix(rTo[:3,:3]).as_quat()
        # print("State:", np.array([rTo[0,3], rTo[1,3], rTo[2,3], quat[0], quat[1], quat[2], quat[3]]))
        return np.array([rTo[0,3], rTo[1,3], rTo[2,3], quat[0], quat[1], quat[2], quat[3]])


    def get_arm_joint_angles(self):
        joint_positions = self.ur5e_sm.get_joint_positions()

        return joint_positions[0:6]
    
    
    def step(self, action):
        dist = self._bound(action[0], 0, self._max_dist)
        theta = self._bound(action[1], -self._max_base_turn, self._max_base_turn)
        
        self._move_robot(dist, theta)

        self.world.step(render=self.cfg.mdp.render)


    def stop(self):
        # simulation_app.close()
        pass


    def move_robot_base_to_pose(self, x, y, theta):
        orientation_rpy = [0, 0, theta]
        base_q = Rotation.from_euler('xyz', orientation_rpy, degrees=False).as_quat()         

        current_robot_base_pose = self.robot_base_prim.get_world_pose()
        current_robot_arm_base_pose = self.ur5e_prim.get_world_pose()

        aTb = self._get_transformation_matrix(current_robot_arm_base_pose, current_robot_base_pose)
        wTb_1 = self._pose_to_transformation_matrix([x, y, self.robot_base_initial_pose[0][2], base_q[0], base_q[1], base_q[2], base_q[3]])

        aTw = np.matmul(aTb, np.linalg.inv(wTb_1))
        wTa = np.linalg.inv(aTw)
        arm_q = Rotation.from_matrix(wTa[:3,:3]).as_quat()

        self.robot_base_prim.set_world_pose(position = np.array([x, y, self.robot_base_initial_pose[0][2]]),
                                      orientation = np.array([base_q[3], base_q[0], base_q[1], base_q[2]]))

        self.ur5e_prim.set_world_pose(position = np.array([wTa[0][3], wTa[1][3], self.ur5e_initial_pose[0][2]]),
                                      orientation = np.array([arm_q[3], arm_q[0], arm_q[1], arm_q[2]]))

    
    def move_obj_to_pose(self, x, y, theta):
        orientation_rpy = [0, 0, theta]
        obj_q = Rotation.from_euler('xyz', orientation_rpy, degrees=False).as_quat()         

        current_obj_pose = self.object_prim.get_world_pose()

        self.object_prim.set_world_pose(position = np.array([x, y, self.object_initial_pose[0][2]]),
                                      orientation = np.array([obj_q[3], obj_q[0], obj_q[1], obj_q[2]]))

    
    def _move_robot(self, dist, theta, j1 = ur5e_default_joint_angles[0], 
                                       j2 = ur5e_default_joint_angles[1],
                                       j3 = ur5e_default_joint_angles[2],
                                       j4 = ur5e_default_joint_angles[3],
                                       j5 = ur5e_default_joint_angles[4],
                                       j6 = ur5e_default_joint_angles[5]
                    ):
        current_robot_base_pose = self.robot_base_prim.get_world_pose()
        current_robot_arm_base_pose = self.ur5e_prim.get_world_pose()
        
        # NOTE: NVIDIA quaternion format [w,x,y,z]
        quat_base = current_robot_base_pose[1]
        curr_base_theta = Rotation.from_quat([quat_base[1], quat_base[2], quat_base[3], 
            quat_base[0]]).as_euler('xyz', degrees=False)[2]

        # NOTE: first forward motion and then turn
        new_base_x = current_robot_base_pose[0][0] + (dist * np.cos(curr_base_theta))
        new_base_y = current_robot_base_pose[0][1] + (dist * np.sin(curr_base_theta))
        new_base_theta = self._wrap_angle(curr_base_theta + theta)

        self.move_robot_base_to_pose(new_base_x, new_base_y, new_base_theta)

        # angles should be in degrees
        self.move_arm(j1, j2, j3, j4, j5, j6)

        self.world.step(render=self.cfg.mdp.render)


    def _wrap_angle(self, angle):
        angle = math.fmod(angle, 2 * np.pi)
        if (angle >= np.pi):
            angle -= 2 * np.pi
        elif (angle <= -np.pi):
            angle += 2 * np.pi
        return angle


    def _isaac_pose_to_transformation_matrix(self, pose):
        r = Rotation.from_quat([pose[1][1], pose[1][2], pose[1][3], pose[1][0]])
        T = np.identity(4)
        T[:3,:3] = r.as_matrix()
        T[0,3] = pose[0][0] 
        T[1,3] = pose[0][1] 
        T[2,3] = pose[0][2]
        return T


    def _pose_to_transformation_matrix(self, pose):
        r = Rotation.from_quat([pose[3], pose[4], pose[5], pose[6]])
        T = np.identity(4)
        T[:3,:3] = r.as_matrix()
        T[0,3] = pose[0] 
        T[1,3] = pose[1] 
        T[2,3] = pose[2]
        return T


    def _pose_to_isaac_pose(self, pose):
        return (np.array([pose[0], pose[1], pose[2]]), np.array([pose[6], pose[3], pose[4], pose[5]]))


    def _isaac_pose_to_pose(self, pose):
        return np.array([pose[0][0], pose[0][1], pose[0][2], pose[1][1], pose[1][2], pose[1][3], pose[1][0]])


    def _get_transformation_matrix(self, pose_s, pose_d):
        wTs = self._isaac_pose_to_transformation_matrix(pose_s)
        wTd = self._isaac_pose_to_transformation_matrix(pose_d)
        sTw = np.linalg.inv(wTs)
        sTd = np.matmul(sTw, wTd)
        return sTd


    def _deg_to_rad(self, angle):
        return angle * np.pi/180.0


    def _get_reward(self):
        reward = 0
        goal_status = False
        return reward, goal_status


    def reset_arm(self):
        self.ur5e_sm.set_joint_positions(positions=np.deg2rad(self.ur5e_default_joint_angles), joint_indices=np.array([0,1,2,3,4,5]))


    def move_arm(self, w1, w2, w3, w4, w5, w6):
        a1 = self._get_valid_joint_angle(w1, 1)
        a2 = self._get_valid_joint_angle(w2, 2)
        a3 = self._get_valid_joint_angle(w3, 3)
        a4 = self._get_valid_joint_angle(w4, 4)
        a5 = self._get_valid_joint_angle(w5, 5)
        a6 = self._get_valid_joint_angle(w6, 6)
        self.ur5e_sm.set_joint_positions(positions=np.array([a1,a2,a3,a4,a5,a6]), joint_indices=np.array([0,1,2,3,4,5]))


    # angle in degrees
    def _get_valid_joint_angle(self, angle, joint_no):
        angle = self._deg_to_rad(angle)
        if angle < self.ur5e_joint_lower_limits[joint_no-1]:
            angle = self.ur5e_joint_lower_limits[joint_no-1]
        elif angle > self.ur5e_joint_upper_limits[joint_no-1]:
            angle = self.ur5e_joint_upper_limits[joint_no-1]
        return angle

    
    def _get_grasp_pose(self, obj_pose_world, robot_pose_world, x_off, y_off, z_off, x_rot, y_rot, z_rot, world=False):
        grasp_pose_o_tran = [x_off, y_off, z_off]
        grasp_pose_o_rot = Rotation.from_euler('xyz', [x_rot, y_rot, z_rot]).as_quat()
        grasp_pose_o = np.hstack((grasp_pose_o_tran, grasp_pose_o_rot))

        oTg = self._pose_to_transformation_matrix(grasp_pose_o)  

        wTo = self._isaac_pose_to_transformation_matrix(obj_pose_world)
        wTg = np.matmul(wTo, oTg)

        wTr = self._isaac_pose_to_transformation_matrix(robot_pose_world)    # ur5e base frame
        rTw = np.linalg.inv(wTr)
        rTg = np.matmul(rTw, wTg)
        

        grasp_pose = None
        if world:
            quat = Rotation.from_matrix(wTg[:3,:3]).as_quat()
            grasp_pose = np.array([wTg[0,3], wTg[1,3], wTg[2,3], quat[0], quat[1], quat[2], quat[3]]) 
        else:
            quat = Rotation.from_matrix(rTg[:3,:3]).as_quat()
            grasp_pose = np.array([rTg[0,3], rTg[1,3], rTg[2,3], quat[0], quat[1], quat[2], quat[3]]) 

        return grasp_pose

    
    def get_grasp_poses(self, obj_pose_world, robot_pose_world, world=False):
        grasp_pose = self._get_grasp_pose(obj_pose_world, robot_pose_world, 0,0,0.3,0,np.pi,0, world=world)
        self._visualize_grasp_pose(grasp_pose, robot_pose_world, self.grasp_marker_prim, world=world)
        return [grasp_pose]


    def compute_grasp_execution_time(self, obj_pose, robot_arm_pose, visualize=False):
        # For computing new start pose
        ee_position, ee_rot_mat = self.articulation_kinematics_solver.compute_end_effector_pose()
        start_pos = ee_position
        start_rot = Rotation.from_matrix(ee_rot_mat).as_quat()
        # print("End effector pose:", start_pos, start_rot)

        grasp_poses = self.get_grasp_poses(obj_pose, robot_arm_pose)

        grasp_execution_time = 0

        for grasp_pose in grasp_poses:
            grasp_pose_t = self._pose_to_transformation_matrix(grasp_pose)
            task_space_position_targets = np.array([
                [start_pos[0], start_pos[1], start_pos[2]],
                [grasp_pose[0], grasp_pose[1], grasp_pose[2]]
                ])
            task_space_orientation_targets = np.array([
                [start_rot[3], start_rot[0], start_rot[1], start_rot[2]],
                [grasp_pose[6], grasp_pose[3], grasp_pose[4], grasp_pose[5]]
                ])

            try:
                trajectory = self.task_space_trajectory_generator.compute_task_space_trajectory_from_points(
                    task_space_position_targets, task_space_orientation_targets, "tool0"
                )

                if trajectory:
                    grasp_execution_time = trajectory.end_time

                    self.curr_ik_solution = trajectory.get_joint_targets(grasp_execution_time)[0][0:6]

                    if visualize:
                        final_joint_config = trajectory.get_joint_targets(trajectory.end_time)[0]
                        self.robot.set_joint_positions(final_joint_config[0:6], joint_indices=np.array([0,1,2,3,4,5]))
                        self.world.step(render=self.cfg.mdp.render)
            except:
                print("Error in computing trajectory") 

        return grasp_execution_time

    
    def _visualize_grasp_pose(self, grasp_pose, robot_pose_world, prim, world=False):
        if world:
            wTg = self._pose_to_transformation_matrix(grasp_pose)
            quat = Rotation.from_matrix(wTg[:3,:3]).as_quat()
            prim.set_world_pose(position = np.array([wTg[0,3], wTg[1,3], wTg[2,3]]),
                                orientation = np.array([quat[3], quat[0], quat[1], quat[2]]))
        else:
            wTr = self._isaac_pose_to_transformation_matrix(robot_pose_world)
            rTg = self._pose_to_transformation_matrix(grasp_pose)
            wTg = np.matmul(wTr, rTg)
            quat = Rotation.from_matrix(wTg[:3,:3]).as_quat()
            prim.set_world_pose(position = np.array([wTg[0,3], wTg[1,3], wTg[2,3]]),
                                    orientation = np.array([quat[3], quat[0], quat[1], quat[2]]))


    def _create_rigid_body(self, bodyType, boxActorPath, mass, scale, position, rotation, color):
        p = Gf.Vec3f(position[0], position[1], position[2])
        orientation = Gf.Quatf(rotation[0], rotation[1], rotation[2], rotation[3])
        scale = Gf.Vec3f(scale[0], scale[1], scale[2])

        bodyGeom = bodyType.Define(self._stage, boxActorPath)
        bodyPrim = self._stage.GetPrimAtPath(boxActorPath)
        bodyGeom.AddTranslateOp().Set(p)
        bodyGeom.AddOrientOp().Set(orientation)
        bodyGeom.AddScaleOp().Set(scale)
        bodyGeom.CreateDisplayColorAttr().Set([color])

        if mass > 0:
            massAPI = UsdPhysics.MassAPI.Apply(bodyPrim)
            massAPI.CreateMassAttr(mass)
        return bodyGeom


    def _set_up_scene(self) -> None:
        assets_root_path = get_assets_root_path()

        # TODO: make this path relative
        scene_path = self.cfg.env_file
        
        open_stage(usd_path=scene_path)

        self._usd_context = omni.usd.get_context()
        self._stage = self._usd_context.get_stage()

        self.world = World()

        self.world.scene.add(Robot(prim_path="/World/ur5e", name="ur5e"))

        # initialize isaac prims for actors in the scene
        
        self.ur5e_prim = XFormPrim(prim_path="/World/ur5e", name="ur5e")
        self.ur5e_initial_pose = self.ur5e_prim.get_world_pose()
        self.ur5e_curr_pose = self.ur5e_prim.get_world_pose()

        gripper = ParallelGripper(
            end_effector_prim_path="/World/ur5e/onrobot_rg6_base_link",
            joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=np.array([0.628, -0.628]),
            action_deltas=np.array([-0.628, 0.628]),
        )

        self.ur5e_sm = self.world.scene.add(SingleManipulator(prim_path="/World/ur5e", name="ur5e_sm", \
            end_effector_prim_name="tool0", gripper=gripper))

        self.robot = Articulation("/World/ur5e")


        self.robot_base_prim = XFormPrim(prim_path="/World/Base", name="Base")
        self.robot_base_initial_pose = self.robot_base_prim.get_world_pose()
        self.robot_base_pose = self.robot_base_prim.get_world_pose()

        self.robot_base_top_prim = XFormPrim(prim_path="/World/BaseTop", name="BaseTop")
        self.robot_base_top_initial_pose = self.robot_base_top_prim.get_world_pose()
        self.robot_base_top_pose = self.robot_base_top_prim.get_world_pose()

        self.object_prim = XFormPrim(prim_path="/World/_03_cracker_box", name="_03_cracker_box")
        self.object_initial_pose = self.object_prim.get_world_pose()
        self.object_pose = self.object_prim.get_world_pose()

        self.camera_prim = XFormPrim(prim_path="/World/ur5e/tool0/Camera", name="Camera")
        self.camera_initial_pose = self.camera_prim.get_world_pose()
        self.camera_pose = self.camera_prim.get_world_pose()

        self.tool0_prim = XFormPrim(prim_path="/World/ur5e/tool0", name="tool0")
        self.tool0_initial_pose = self.tool0_prim.get_world_pose()
        self.tool0_pose = self.tool0_prim.get_world_pose()

        self.table_prim = XFormPrim(prim_path="/World/Cylinder", name="Cylinder")
        self.table_initial_pose = self.table_prim.get_world_pose()
        self.table_pose = self.table_prim.get_world_pose()

        self.grasp_marker = self._create_rigid_body(
            UsdGeom.Cone,
            "/GraspMarker3",
            0,
            [0.03, 0.03, 0.03],
            [0,0,0],
            [1,0,0,0],
            color=Gf.Vec3f([0,1.0,0]),
        )
        self.grasp_marker_prim = XFormPrim(prim_path="/GraspMarker3", name="GraspMarker3")


        # Lula Kinematics solver
        self.lula_kinematics_solver = LulaKinematicsSolver(
            robot_description_path = self.cfg.robot_description_file,
            urdf_path = self.cfg.urdf_file
        )

        self.articulation_kinematics_solver = ArticulationKinematicsSolver(self.robot, self.lula_kinematics_solver, "tool0")

        # Initialize a LulaCSpaceTrajectoryGenerator object
        self.task_space_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path = self.cfg.robot_description_file,
            urdf_path = self.cfg.urdf_file
        )

        # start simulation
        self.world.reset()

        # Initialization should happen only after simulation starts
        self.robot.initialize()

        print("Simulation start status: ", self.world.is_playing())

        if self.world.is_playing():
            pass

        self.reset()