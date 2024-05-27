from pre_grasp_approaching.tasks.task import Task

# import ikfastpy
import numpy as np
import datetime
import math
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, Point
import time

from omni.isaac.kit import SimulationApp
from pxr import UsdLux, UsdGeom, Sdf, Gf, UsdPhysics
from omni.isaac.core.prims import XFormPrim
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from omni.isaac.motion_generation import ArticulationTrajectory # not available in old Isaac 1.1
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation.lula import LulaTaskSpaceTrajectoryGenerator # not available in old Isaac 1.1
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper


class BaseMotion(Task):
    """
    Approaching base motion
    """

    def __init__(
        self ,
        cfg
    ) -> None:
        self.cfg = cfg
        self._last_dist_arm_to_obj = 2.5
        self._no_of_ik_solutions = 0
        self._attempt_manipulation = 0
        self._manipulation_attempted = 0
        self._selected_grasp_pose = np.zeros((2,))
        self._stop = 0
        self._stopped = 0

        self._base_poses = np.array([])
        self._obj_poses = np.array([])
        self._arm_poses = np.array([])

        super().__init__(cfg)


    
    def reset(self, state=None):
        self._last_dist_arm_to_obj = 2.5
        self._no_of_ik_solutions = 0

        self._selected_grasp_pose = np.zeros((2,))
        self._stopped = 0

        return super().reset(robot_r_min=1.5, robot_r_max=2.0)

    
    def get_error(self, grasp_pose_gt, tool0_pose_gt):
        grasp_pose_r = Rotation.from_quat(grasp_pose_gt[3:7]).as_matrix()
        tool0_pose_r = Rotation.from_quat(tool0_pose_gt[3:7]).as_matrix()

        error_cos = 0.5 * (np.trace(grasp_pose_r.dot(np.linalg.inv(tool0_pose_r))) - 1.0)
        error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
        error = math.acos(error_cos)
        r_error = 180.0 * error / np.pi # [rad] -> [deg]

        x_error = np.power(grasp_pose_gt[0] - tool0_pose_gt[0], 2)
        y_error = np.power(grasp_pose_gt[1] - tool0_pose_gt[1], 2)
        z_error = np.power(grasp_pose_gt[2] - tool0_pose_gt[2], 2)
        t_error = np.power(x_error + y_error + z_error, 0.5)
        
        return t_error, r_error

    
    def step(self, action):
        self._no_of_ik_solutions = 0

        dist = self._bound(action[0], 0, self._max_dist)
        theta = self._bound(action[1], -self._max_base_turn, self._max_base_turn)
        
        self._move_robot(dist, theta)

        self._state = self._get_state()
        self._n_steps = self._n_steps + 1

        reward, goal_status = self._get_reward()
        return self._state, reward, goal_status, {}

    
    def check_collision(self):
        robot_pose = self.robot_base_prim.get_world_pose()
        obj_pose = self.object_prim.get_world_pose()

        robot_rot_rpy = Rotation.from_quat([robot_pose[1][1], robot_pose[1][2], robot_pose[1][3], robot_pose[1][0]]).as_euler('xyz')

        radius = 0.4 + 0.0
        table_polygon = Point(obj_pose[0][0], obj_pose[0][1]).buffer(radius)

        rl = 0.8/2
        rw = 0.5/2
        r = pow(pow(rl,2) + pow(rw,2), 0.5)
        ang = math.atan(rl/rw) 
        r_off = 0

        robot_c1 = [robot_pose[0][0] + (r*np.cos(self._wrap_angle(robot_rot_rpy[2] + ang + r_off))), robot_pose[0][1] + (r*np.sin(self._wrap_angle(robot_rot_rpy[2] + ang + r_off))) ]
        robot_c2 = [robot_pose[0][0] + (r*np.cos(self._wrap_angle(robot_rot_rpy[2] - ang + r_off))), robot_pose[0][1] + (r*np.sin(self._wrap_angle(robot_rot_rpy[2] - ang + r_off))) ]
        robot_c3 = [robot_pose[0][0] + (r*np.cos(self._wrap_angle(robot_rot_rpy[2] + ang - np.pi + r_off))), robot_pose[0][1] + (r*np.sin(self._wrap_angle(robot_rot_rpy[2] + ang - np.pi + r_off)))]
        robot_c4 = [robot_pose[0][0] + (r*np.cos(self._wrap_angle(robot_rot_rpy[2] - ang + np.pi + r_off))), robot_pose[0][1] + (r*np.sin(self._wrap_angle(robot_rot_rpy[2] - ang + np.pi + r_off)))]
        
        robot = [robot_c1, robot_c2, robot_c3, robot_c4, robot_c1]
        robot_polygon = Polygon(robot)

        collision_status = table_polygon.intersects(robot_polygon) or table_polygon.contains(robot_polygon) or robot_polygon.contains(table_polygon)
        return collision_status

    
    def get_no_of_ik_solutions_lula(self, obj_pose, robot_arm_pose, visualize=False):
        grasp_poses = self.get_grasp_poses(obj_pose, robot_arm_pose, world=True)
        no_of_ik_solutions = 0
        for grasp_pose in grasp_poses:
            grasp_pose_t = self._pose_to_transformation_matrix(grasp_pose)

            # For Lula
            self.lula_kinematics_solver.set_robot_base_pose(robot_arm_pose[0], robot_arm_pose[1])

            ik = self.lula_kinematics_solver.compute_inverse_kinematics("tool0", grasp_pose[0:3], 
                np.array([grasp_pose[6], grasp_pose[3], grasp_pose[4], grasp_pose[5]]))

            if ik[1]:
                no_of_ik_solutions = no_of_ik_solutions + 1

                if visualize:
                    self.ur5e_sm.set_joint_positions(positions=ik[0][0:6], joint_indices=np.array([0,1,2,3,4,5]))
                    self.world.step(render=True)

        return no_of_ik_solutions
            

    def _get_reward(self):
        reward = 0
        reward_time_penalty = -1000

        goal_status = False   # is episode over

        object_pose = self.object_prim.get_world_pose()
        robot_base_pose = self.robot_base_prim.get_world_pose()
        robot_arm_pose = self.ur5e_prim.get_world_pose()

        # object table radius: 40 cm (can be used for checking collision)
        # robot max radius: 40 cm
        # collision_dist = 0.4 + 0.4
        # bb_area = 0

        dist_base_to_obj = np.sqrt(np.power(object_pose[0][0] - robot_base_pose[0][0],2) + 
                        np.power(object_pose[0][1] - robot_base_pose[0][1],2))

        dist_arm_to_obj = np.sqrt(np.power(object_pose[0][0] - robot_arm_pose[0][0],2) + 
                np.power(object_pose[0][1] - robot_arm_pose[0][1],2))


        if self.check_collision():
            reward = reward -100000
            goal_status = True
            return reward, goal_status


        reward = reward_time_penalty

        self._no_of_ik_solutions = self.get_no_of_ik_solutions_lula(object_pose, robot_arm_pose, visualize=False)
                    
        if self._no_of_ik_solutions > 0:
            reward =  (self._no_of_ik_solutions * 100000) + reward
                            

        if self._n_steps > 24:
            # save data for analysis

            # self._base_poses = np.append(self._base_poses, self._isaac_pose_to_pose(robot_base_pose))
            # self._obj_poses = np.append(self._obj_poses, self._isaac_pose_to_pose(object_pose))
            # self._arm_poses = np.append(self._arm_poses, self._isaac_pose_to_pose(robot_arm_pose))

            # np.savez('{}/data_for_analysis.npz'.format('/home/sdur/'), full_save=True, 
            #     base_poses=self._base_poses, obj_poses=self._obj_poses,
            #     arm_poses=self._arm_poses)

            goal_status = True

        return reward, goal_status
