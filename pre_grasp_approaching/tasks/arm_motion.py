from motion_planning.tasks.task import Task

import ikfastpy
import numpy as np
import datetime
import math
import time
from scipy.spatial.transform import Rotation
import torch

from omni.isaac.kit import SimulationApp
from pxr import UsdLux, UsdGeom, Sdf, Gf, UsdPhysics
from omni.isaac.core.prims import XFormPrim

from mushroom_rl.utils import spaces
from mushroom_rl.core import Environment, MDPInfo

from train.state_prediction import MLP



class ArmMotion(Task):
    """
    Task for learning arm motion
    """

    def __init__(
        self ,
        cfg
    ) -> None:

        self.cfg = cfg
        self._no_of_ik_solutions = 0
        self._selected_grasp_pose = -1
        self._joint_movements = np.zeros((3,))
        self._last_dist_arm_to_obj = 2.5
        self.curr_grasp_exec_time  = 0
      
        observation_space = spaces.Box(low=np.array([-3,-3,-3,-1,-1,-1,-1, -2*np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]), 
                                        high=np.array([3,3,3,1,1,1,1, 2*np.pi, 0, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]))
        
        self._max_shoulder_pan_turn = self.cfg.mdp.max_shoulder_pan_turn
        self._max_shoulder_lift_turn = self.cfg.mdp.max_shoulder_lift_turn
        self._max_elbow_turn = self.cfg.mdp.max_elbow_turn

        action_space = spaces.Box(low=np.array([-self._max_shoulder_pan_turn, -self._max_shoulder_lift_turn, -self._max_elbow_turn]),
                                  high=np.array([self._max_shoulder_pan_turn, self._max_shoulder_lift_turn, self._max_elbow_turn]))
        
        mdp_info = MDPInfo(cfg)

        self.initialize_mdp(mdp_info)
        self.initialize_parameters()

        # initializes the episode scene
        self._set_up_scene()

        # load state predictor
        self.state_predictor = MLP(3,3)
        self.state_predictor.load_state_dict(torch.load(cfg.train.bp_net))

        self.get_arm_joint_angles()


    
    def reset(self, state=None):
        self._no_of_ik_solutions = 0
        self._selected_grasp_pose = -1
        self._joint_movements = np.zeros((3,))
        self._curr_ik = np.zeros((6,))
        self._curr_pg_ik = np.zeros((6,))
        self._curr_ik_status = False
        self.curr_grasp_exec_time = 0

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


    def get_arm_joint_angles(self):
        joint_positions = super().get_arm_joint_angles()
        self._shoulder_pan_angle, self._shoulder_lift_angle, self._elbow_angle = joint_positions[0], joint_positions[1], joint_positions[2]
        return joint_positions


    def _get_state(self):
        obj_pose_in_robot_frame = super()._get_state()
        arm_joint_angles = self.get_arm_joint_angles()
        return np.hstack((obj_pose_in_robot_frame, arm_joint_angles))

    
    def step(self, action):
        # [dist(base), theta(base), grasp, dont grasp, shoulder_pan, shoulder_lift, elbow]
        self._no_of_ik_solutions = 0

        dist = action[0]
        theta = action[1]

        shoulder_pan_turn = self._bound(action[4], -self._max_shoulder_pan_turn, self._max_shoulder_pan_turn)
        shoulder_lift_turn = self._bound(action[5], -self._max_shoulder_lift_turn, self._max_shoulder_lift_turn)
        elbow_turn = self._bound(action[6], -self._max_elbow_turn, self._max_elbow_turn)

        self._joint_movements = self._joint_movements + np.absolute(np.array([shoulder_pan_turn, shoulder_lift_turn, elbow_turn]))
        
        shoulder_pan_angle = self._wrap_angle(self._shoulder_pan_angle + shoulder_pan_turn)
        shoulder_lift_angle = self._wrap_angle(self._shoulder_lift_angle + shoulder_lift_turn)
        elbow_angle = self._wrap_angle(self._elbow_angle + elbow_turn)


        self._move_robot(dist, theta, j1=np.rad2deg(shoulder_pan_angle), j2=np.rad2deg(shoulder_lift_angle), \
            j3=np.rad2deg(elbow_angle))

        self._attempt_manipulation = action[2]

        self._state = self._get_state()
        self._n_steps = self._n_steps + 1

        reward, goal_status = self._get_reward()        
        return self._state, reward, goal_status, {}

    
    def _get_predicted_terminal_state(self, state):
        obj_pose = self.object_prim.get_world_pose()
        wTo = self._isaac_pose_to_transformation_matrix(obj_pose)

        base_init = np.zeros((3,))
        base_init_norm = np.zeros((3,))

        base_init[0:2] = state[0:2]
        base_init[2] = Rotation.from_quat(state[3:7]).as_euler('xyz')[2]

        # normalization
        base_init_norm[0] = base_init[0]/6.0
        base_init_norm[1] = base_init[1]/6.0
        base_init_norm[2] = base_init[2]/6.0

        ip = torch.from_numpy(base_init_norm).float()
        op = self.state_predictor.forward(ip)
        op = op.cpu().detach().numpy()

        # denormalization
        prediction = np.zeros((7,))
        prediction[0] = op[0]*6
        prediction[1] = op[1]*6
        prediction[2] = -0.06038878    # offset should be changed based on the model
        prediction[3:7] = Rotation.from_euler('xyz', [0, 0, op[2]*np.pi]).as_quat()
        rpTo = self._pose_to_transformation_matrix(prediction)
        oTrp = np.linalg.inv(rpTo)
        wTrp = np.matmul(wTo, oTrp)
        
        quat = Rotation.from_matrix(wTrp[:3,:3]).as_quat()
        robot_pose_in_world = np.array([wTrp[0,3], wTrp[1,3], wTrp[2,3], quat[0], quat[1], quat[2], quat[3]])
        return robot_pose_in_world


    def get_euclidean_distance(self, pose1, pose2):
        return np.power((pose1[0][0]-pose2[0][0])**2 + (pose1[0][1]-pose2[0][1])**2 + (pose1[0][2]-pose2[0][2])**2 , 0.5)


    def compute_ik_base_reward(self, obj_pose, robot_arm_pose, curr_joint_positions, visualize=False):
        grasp_poses = self.get_grasp_poses(obj_pose, robot_arm_pose, world=True)
        max_diff = 0
        reward = 0
        
        grasp_pose = grasp_poses[0]
        pre_grasp_pose = grasp_poses[0].copy()
        pre_grasp_pose[2] = pre_grasp_pose[2] + 0.3

        grasp_pose_t = self._pose_to_transformation_matrix(grasp_pose)

        # For Lula
        self.lula_kinematics_solver.set_robot_base_pose(robot_arm_pose[0], robot_arm_pose[1])

        ik = self.lula_kinematics_solver.compute_inverse_kinematics("tool0", grasp_pose[0:3], 
            np.array([grasp_pose[6], grasp_pose[3], grasp_pose[4], grasp_pose[5]]))


        if ik[1]:
            ik_pi_to_pi = np.zeros((3,))
            curr_joints_pi_to_pi = np.zeros((3,))

            ik_pi_to_pi[0] = self._wrap_angle(ik[0][0])
            ik_pi_to_pi[1] = self._wrap_angle(ik[0][1])
            ik_pi_to_pi[2] = self._wrap_angle(ik[0][2])

            curr_joints_pi_to_pi[0] = self._wrap_angle(curr_joint_positions[0])
            curr_joints_pi_to_pi[1] = self._wrap_angle(curr_joint_positions[1])
            curr_joints_pi_to_pi[2] = self._wrap_angle(curr_joint_positions[2])

            joint_diffs = np.absolute(ik_pi_to_pi - curr_joints_pi_to_pi)

            max_diff = np.max(joint_diffs[0:3])

            self._curr_ik = ik[0][0:6]
            self._curr_ik_status = True

            if visualize:
                self.ur5e_sm.set_joint_positions(positions=ik[0][0:6], joint_indices=np.array([0,1,2,3,4,5]))
                self.world.step(render=True)

            reward = 1.0/(max_diff + 1)
            
        return reward

    
    def _get_reward(self):
        reward = 0
        goal_status = False   # is episode over
        obj_pose = self.object_prim.get_world_pose()
        tool_pose = self.tool0_prim.get_world_pose()
        robot_pose = self.ur5e_prim.get_world_pose()
        robot_base_pose = self.robot_base_prim.get_world_pose()

        dist_base_to_obj = np.sqrt(np.power(obj_pose[0][0] - robot_base_pose[0][0],2) + 
                        np.power(obj_pose[0][1] - robot_base_pose[0][1],2))

        dist_arm_to_obj = np.sqrt(np.power(obj_pose[0][0] - robot_pose[0][0],2) + 
                np.power(obj_pose[0][1] - robot_pose[0][1],2))

        curr_joint_positions = self.get_arm_joint_angles()

        reward_distance = 5000 * (2.5 - dist_arm_to_obj)/2.5

        reward_dist_diff = 1000 * (self._last_dist_arm_to_obj - dist_arm_to_obj)/2.5
        self._last_dist_arm_to_obj = dist_arm_to_obj

        navigation_time = self._n_steps

        if self._attempt_manipulation:
            print("Terminating ....")

            reward = self.compute_ik_base_reward(obj_pose, robot_pose, curr_joint_positions, visualize=True) * 100000

            # grasp_execution_time =  self.compute_grasp_execution_time(obj_pose, robot_pose, visualize=False)
            # print("Attempting manipulation:", self._n_steps, "Execution time: ", grasp_execution_time)

            self.curr_grasp_exec_time =  0 #grasp_execution_time

            # if grasp_execution_time > 0:
            #     reward = reward + (100000/(grasp_execution_time+1)) + 0000

            goal_status = True
        else:
            pred_robot_pose = self._pose_to_isaac_pose(self._get_predicted_terminal_state(self._state))
            reward = self.compute_ik_base_reward(obj_pose, pred_robot_pose, curr_joint_positions) * 50000

            grasp_execution_time =  self.compute_grasp_execution_time(obj_pose, pred_robot_pose, visualize=False)
            if grasp_execution_time > 0:
                reward = reward + (1000/(grasp_execution_time+1))

        if self._n_steps > 24:
            goal_status = True

        return reward, goal_status