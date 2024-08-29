import multiprocessing
import numpy as np
import math as m
import scipy.spatial as spsp

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MortarObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
import robosuite.utils.transform_utils as T
from ur_control.traj_utils import compute_trajectory

from ur_control import spalg
import rospkg
import json

# Default Grind environment configuration
DEFAULT_GRIND_CONFIG = {
    # settings for reward
    "reward_weights": {
        "tracking_trajectory_error": 1.0,  # reward for following the trajectory reference
        "tracking_force_error": 1.0,  # reward for pushing into the mortar according to te force reference
    },
    "task_complete_reward": 50.0,  # reward per task done
    "exit_task_space_penalty": 0,  # penalty for moving too far away from the mortar task space
    "collision_penalty": 0,  # reward for increased velocity
    "excess_force_penalty": 0,  # penalty for each step that the force is over the safety threshold

    "force_follow_normalization": [50.0, 50.0, 50.0, 10.0, 10.0, 10.0],  # max load (N)
    "traj_follow_normalization": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],  # max distance between waypoints

    # settings for thresholds
    "force_torque_limits": [50.0, 50.0, 50.0, 10.0, 10.0, 10.0],  # maximum eef force/torque allowed (N | N/m)
    "mortar_space_threshold_max": 0.1,  # maximum distance from the mortar the eef is allowed to diverge (m)

    # tracking settings
    "tracking_trajectory_threshold": 0.01,
    "tracking_trajectory_method": 'per_error_threshold',

    # Task settings
    "mortar_height": 0.047,  # (m)
    "mortar_max_radius": 0.04,  # (m)

    # misc settings
    "evaluate": False,
    "print_results": False,  # Whether to print results or not
    "log_rewards": False,
    "log_details": True,
    "log_dir": "log",
    "get_info": False,  # Whether to grab info after each env step if not
    "use_robot_obs": True,  # if we use robot observations (proprioception) as input to the policy
    "early_terminations": True,  # Whether we allow for early terminations or not
}


TRACKING_METHODS = [
    'per_step',  # update waypoints after every step
    'per_error_threshold'  # update waypoints after the tracking error is smaller than `tracking_trajectory_threshold`
]


class OSXGrind(SingleArmEnv):
    """
    This class corresponds to the grinding task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. For this environment, setting a value other than the default ("Grinder")
            will raise an AssertionError, as this environment is not meant to be used with any other alternative gripper.

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (mortar) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

        task_config (None or dict): Specifies the parameters relevant to this task. For a full list of expected
            parameters, see the default configuration dict at the top of this file.
            If None is specified, the default configuration will be used.

    Raises:
        AssertionError: [Gripper specified]
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="Grinder",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        task_config=DEFAULT_GRIND_CONFIG,
        reference_trajectory=None,
        reference_force=None,
        action_indices=range(0, 6)
    ):

        # Assert that the gripper type is Grinder
        assert (
            gripper_types == "Grinder"
        ), "Tried to specify gripper other than Grinder in Grind environment!"

        # self.log_details = log_details
        self.horizon = horizon

        self.task_config = task_config

        self.print_results = self.task_config["print_results"]
        self.log_rewards = self.task_config["log_rewards"]
        self.log_details = self.task_config["log_details"]
        self.early_terminations = self.task_config["early_terminations"]

        self.force_follow_normalization = self.task_config["force_follow_normalization"]
        self.traj_follow_normalization = np.array(self.task_config["traj_follow_normalization"])

        self.force_torque_limits = self.task_config['force_torque_limits']

        # settings for the reward
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.reward_weights = self.task_config['reward_weights']
        self.exit_task_space_penalty = self.task_config["exit_task_space_penalty"]
        self.task_complete_reward = self.task_config["task_complete_reward"]
        self.collision_penalty = self.task_config["collision_penalty"]
        self.excess_force_penalty = self.task_config["excess_force_penalty"]

        # settings for table top and task space
        self.mortar_height = self.task_config["mortar_height"]
        self.mortar_radius = self.task_config["mortar_max_radius"]
        self.mortar_space_threshold_max = self.task_config["mortar_space_threshold_max"]
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.task_box = np.array([self.mortar_radius, self.mortar_radius, self.mortar_height+self.table_offset[2]]) + self.mortar_space_threshold_max

        # references to follow
        self.current_waypoint_index = 0
        self.ft_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Add an extra waypoint to make sure that every waypoint is
        # tracked before considering the tracking completed
        reference_trajectory = np.append(reference_trajectory, [reference_trajectory[-1]], axis=0)
        self.reference_trajectory = reference_trajectory
        self.trajectory_len = len(self.reference_trajectory)
        self.reference_force = reference_force
        self.tracking_trajectory_threshold = self.task_config['tracking_trajectory_threshold']
        self.tracking_trajectory_method = self.task_config['tracking_trajectory_method']
        # Verify the proposed impedance mode is supported
        assert self.tracking_trajectory_method in TRACKING_METHODS, (
            "Error: unsupported tracking method"
            "Inputted tracking method: {}, Supported methods: {}".format(self.tracking_trajectory_method, TRACKING_METHODS)
        )

        # actor action subset
        self.action_indices = action_indices

        # object placement initializer
        self.placement_initializer = placement_initializer

        # log data
        self.prev_quat = self.reference_trajectory[0, 3:]
        self.evaluate = self.task_config['evaluate']
        self.log_dir = self.task_config['log_dir']
        self.log_dict = {
            'rewards': {
                'traj_error': [],
                'force_error': [],
            },
            'details': {
                'timesteps': [],
                'waypoint': [],
                'action_in': [],
                'res_action': [],
                'scl_action': [],
                'current_ref': [],
                'current_pos': [],
                'current_quat': [],
                'current_force_ref': [],
                'kp': [],
                'current_force': [],
                'current_force_ref_eef_frame': [],
                'current_force_eef_frame': [],
            }
        }

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def step(self, action):

        assert len(action) == len(self.action_indices), \
            f"Size of action {len(action)} does not match expected {len(self.action_indices)}"

        action_kp = np.interp(action, [-1,1], [0.001, 0.5]) # limits for kp (action from sac)
        # change controller params
        self.robots[0].controller.kp[self.action_indices] = action_kp[self.action_indices]

        pos_rot_action = np.zeros(6)

        if self.reference_force is not None:
            if self.robots[0].controller.desired_ft_frame == "hand":
                # If i receive a reference in the gripper frame directly, change it to base frame
                gripper_in_robot_base = self.robots[0].controller.pose_in_base_from_name(f"{self.robots[0].controller.ft_prefix}_eef")
                ee_force, ee_torque = T.force_in_A_to_force_in_B(self.reference_force[self.current_waypoint_index][:3],
                                                                self.reference_force[self.current_waypoint_index][3:],
                                                                gripper_in_robot_base) #TODO replace with spalg.convert_wrench 

                self.ft_action = np.concatenate([ee_force, ee_torque])

            else:
                # If i receive a reference in base frame directly
                self.ft_action = self.reference_force[self.current_waypoint_index]

        else:
            # if no force reference given, compute it at each step wrt world frame orientation
            self.ft_action = self.compute_force_ref(self._eef_xpos)

        # update in rendering the cylinder representing the normal force/direction
        # assume orientation is given perpendicular to mortar surface
        self.sim.model.body_pos[self.sim.model.body_name2id("force_cylinder_main")] = self.reference_trajectory[self.current_waypoint_index][:3]-np.array([0, 0, 0.02])
        self.sim.model.body_quat[self.sim.model.body_name2id("force_cylinder_main")] = self.reference_trajectory[self.current_waypoint_index][3:]

        # send action with both pose and wrench to the env
        # send the ft action already in base frame
        ctr_action = np.concatenate([pos_rot_action, self.ft_action])

        # online tracking of the reference trajectory
        residual_action = self._compute_relative_distance()
        factor = 1 - np.tanh(100 * self.tracking_error)
        scale_factor = np.interp(factor, [0.0, 1.0], [1, 80])  # TODO get some good values for per step and per thresh
        ctr_action[:6] += residual_action * scale_factor

        self.__log_details__(action, residual_action)
        return super().step(ctr_action)

    def compute_force_ref(self, eef_pose):

        # Calculating  the force at every step

        # Define the bowl surface function and its derivatives
        def f(xy): return (xy[0]**4 + xy[1]**4) * 11445.39 + (xy[1]**2 * xy[0]**2 * 22890.7) + xy[0]**2 * 3.11558 + xy[1]**2 * 3.11558 - 0.038811
        def fx(x, y): return 4*x**3 * 11445.39 + y**2 * 2*x * 22890.7 + 2*x * 3.11558
        def fy(x, y): return 4*y**3 * 11445.39 + 2*y * x**2 * 22890.7 + 2*y * 3.11558

        def normal_vector(x, y):
            grad = np.array([fx(x, y), fy(x, y), -1])
            return grad / np.linalg.norm(grad)

        # search in vicinity of eef pose xyz TODO change from search to assumption
        XY = np.mgrid[eef_pose[0]-0.03:eef_pose[0]+0.03:0.001, eef_pose[0]-0.03:eef_pose[0]+0.03:0.001].reshape(2, -1).T
        Z = [f(xy) for xy in XY]
        Z = np.array(Z).reshape(-1, 1)
        A = np.concatenate([XY, Z], axis=1)

        # find closest point on the mortar to the tip of the eef and calculate normal to surface in that point
        distance, index = spsp.KDTree(A).query(eef_pose)
        closest_point = A[index]
        normal_vect_direction = normal_vector(closest_point[0], closest_point[1])

        return np.concatenate([normal_vect_direction, np.array([0, 0, 0])])  # is it zeroes or sth else

    def reward(self, action=None):

        reward = 0.0

        # If the arm does not present unwanted behaviors (collisions, reaching joint limits,
        # or other conditions based on @termination_flag), calculate reward
        # (we don't want to reward grinding if there are unsafe situations)

        if not self._check_task_space_limits():
            reward = self.exit_task_space_penalty
            self.task_space_exits += 1
        elif not self._check_force_torque_limits():
            reward = self.excess_force_penalty
            self.f_excess += 1
        elif self.check_contact(self.robots[0].robot_model):
            if self.reward_shaping:
                reward = self.collision_penalty
            self.collisions += 1

        # sparse completion reward
        elif self._check_success():
            reward = self.task_complete_reward

        else:
            # use a shaping reward
            if self.reward_shaping:

                # Reward for pushing into mortar with desired linear forces
                distance_from_ref_force = -self.tracking_force_error
                force_reward = self.reward_weights['tracking_force_error'] * distance_from_ref_force

                # Reward for following desired linear trajectory
                tracking_trajectory_error = -max(self.tracking_error - self.tracking_trajectory_threshold, 0.0) / self.tracking_trajectory_threshold
                # print(self.tracking_error, tracking_trajectory_error)
                traj_reward = self.reward_weights['tracking_trajectory_error'] * tracking_trajectory_error

                # TODO: reward for finishing faster?

                reward += force_reward + traj_reward

                if self.log_rewards:
                    self.log_dict['rewards']['traj_error'].append(traj_reward)
                    self.log_dict['rewards']['force_error'].append(force_reward)

                # Printing results
                if self.print_results:
                    string_to_print = (
                        "Process {pid}, timestep {ts:>4}: reward: {rw:8.4f}, collisions: {sc:>3}, f_excess: {fe:>3}, task_space_limit: {te:>3}".format(
                            pid=id(multiprocessing.current_process()),
                            ts=self.timestep,
                            rw=reward,
                            sc=self.collisions,
                            fe=self.f_excess,
                            te=self.task_space_exits,
                        )
                    )
                    print(string_to_print)

        return reward

    def _compute_relative_distance(self):
        relative_distance = np.zeros(6)
        if self.reference_trajectory is not None:
            relative_distance[:3] = self.reference_trajectory[self.current_waypoint_index][:3] - self._eef_xpos
            ref_quat = self.reference_trajectory[self.current_waypoint_index][3:]
            relative_distance[3:] = spalg.quaternions_orientation_error(ref_quat, self._eef_xquat)

        # track error of the actions controlled by the policy
        self.tracking_error = np.linalg.norm(relative_distance)
        return relative_distance

    def _compute_relative_wrenches(self):
        relative_wrench = np.zeros(6)

        # in base frame
        relative_wrench = self.ft_action - self.ee_wrench

        self.tracking_force_error = np.linalg.norm(relative_wrench / self.force_follow_normalization)
        return relative_wrench

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # (For UR5e) specific starting point
        if self.robots[0].name == "UR5e":
            ros_pack = rospkg.RosPack()
            qpos_init_file = ros_pack.get_path("osx_powder_grinding") + "/config/ur5e_init_qpos.json"
            init_joints = json.load(open(qpos_init_file))["init_q"]
            self.robots[0].init_qpos = np.array(init_joints)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.mortar = MortarObject(
            name="mortar",
        )

        # add the "ref force arrow in rendering"
        cylinder_radius = 0.002
        cylinder_length = 0.02
        self.force_cylinder = CylinderObject(
            name="force_cylinder",
            size=(cylinder_radius, cylinder_length),
            rgba=[0, 1, 0, 1],
            joints=None,
            duplicate_collision_geoms=False,
            obj_type='visual',
        )

        # Load cylinder object
        self.force_cylinder_object = self.force_cylinder.get_obj()
        self.force_cylinder_object.set("pos", "0.1  0.1  0.9")

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.mortar)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.mortar,
                x_range=[-0.0001, 0.0001],
                y_range=[-0.0001, 0.0001],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.00,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.mortar, self.force_cylinder],
        )

        self.model.merge_assets(self.force_cylinder)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.mortar_body_id = self.sim.model.body_name2id(self.mortar.root_body)
        self.force_cylinder_body_id = self.sim.model.body_name2id(self.force_cylinder.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=f"{pf}proprio")
        def robot0_relative_pose(obs_cache):
            return self._compute_relative_distance()/self.traj_follow_normalization

        @sensor(modality=f"{pf}proprio")
        def robot0_relative_wrench(obs_cache):
            return self._compute_relative_wrenches()

        @sensor(modality=f"{pf}proprio")
        def robot0_wrench(obs_cache):
            return self.ee_wrench

        # # needed in the list of observables
        @sensor(modality=f"{pf}proprio")
        def robot0_eef_force(obs_cache):
            return self.ee_wrench[:3]

        @sensor(modality=f"{pf}proprio")
        def robot0_eef_torque(obs_cache):
            return self.ee_wrench[3:]

        sensors = [robot0_eef_force, robot0_eef_torque, robot0_relative_pose, robot0_relative_wrench]
        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # recompute the referece trajectory for each episode, with height dependent on radius
        y = np.array([0.026, 0.027, 0.03, 0.032])
        x = np.array([0.008, 0.01, 0.012, 0.015])
        coefficients = np.polyfit(x, y, deg=2)
        f = np.poly1d(coefficients)

        R = 0.01 * np.random.random_sample() + 0.008 # take a random radius inside the mortar
        self.num_waypoints = np.random.randint(low=100, high=500) # take a random number of points for the trajectory
        reference_trajectory, self.reference_force = self.recompute_trajectory(R, f(R), self.num_waypoints)
        reference_trajectory = np.append(reference_trajectory, [reference_trajectory[-1]], axis=0)
        self.reference_trajectory = reference_trajectory
        self.trajectory_len = len(self.reference_trajectory)

        self.current_waypoint_index = 0
        self.collisions = 0
        self.f_excess = 0
        self.task_space_exits = 0

    def recompute_trajectory(self, R, h, num_waypoints):
        # Compute reference trajectory and reference force profile

        # mortar surface function and derivatives
        def fx(x, y): return 4*x**3 * 11445.39  + y**2 * 2*x * 22890.7 + 2*x * 3.11558
        def fy(x, y): return 4*y**3 * 11445.39  + 2*y * x**2 * 22890.7 + 2*y * 3.11558

        def get_orientation_quaternion_smooth(n, prev_quat=None, prev_R=None):
            if prev_R is None:
                # If no previous rotation
                R = np.zeros((3, 3))
                R[:, 2] = n
                R[:, 0] = np.cross([0, 1, 0], n)
                R[:, 0] /= np.linalg.norm(R[:, 0])
                R[:, 1] = np.cross(n, R[:, 0])
            else:
                # If we have a previous rotation, try to minimize the change
                R = prev_R.copy()
                R[:, 2] = n  # Set the new normal
                R[:, 1] = np.cross(n, R[:, 0])  # Adjust the y-axis
                R[:, 1] /= np.linalg.norm(R[:, 1])
                R[:, 0] = np.cross(R[:, 1], n)  # Adjust the x-axis

            quat = T.mat2quat(R)

            # Ensure consistent quaternion sign
            if prev_quat is not None and np.dot(quat, prev_quat) < 0:
                quat = -quat

            return quat, R

        h = np.clip(h, 0.01, 0.04) # make sure it's still inside the mortar as height
        table_height = 0.8
        initial_pose = [0.0, R, table_height+h, 0.0, 0.9990052, 0.04459406, 0.0]

        ref_traj = compute_trajectory(initial_pose,
                                        plane='XY',
                                        radius=R,
                                        radius_direction='-Y',
                                        steps=num_waypoints,
                                        revolutions=1, from_center=False, trajectory_type='circular')

        ind = 0

        # recalculate orientation in every point to be perpendicular to surface;
        for point in ref_traj:
            # point to evaluate in
            px, py = point[0], point[1]

            # calculate the normal and tangents vectors to the surface of the mortar
            n = np.array([fx(px, py), fy(px, py), -1])
            normal_vect_direction = n/np.linalg.norm(n)

            try:
                quat_ref, R = get_orientation_quaternion_smooth(normal_vect_direction, quat_ref, R)
            except:
                # first point won't have a rotation matrix to refer to
                quat_ref, R = get_orientation_quaternion_smooth(normal_vect_direction)

            ref_traj[ind,3:] = quat_ref
            ind+=1

        load_N = np.random.randint(low=1, high=20) # take a random force reference
        ref_force = np.array([[0, 0, load_N, 0, 0, 0]]*num_waypoints)


        return ref_traj, ref_force

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)

        if self.current_waypoint_index < self.trajectory_len - 1:
            if self.tracking_trajectory_method == 'per_step':
                self.current_waypoint_index += 1

            elif self.tracking_trajectory_method == 'per_error_threshold':
                if self.tracking_error < self.tracking_trajectory_threshold:
                    self.current_waypoint_index += 1

        # allow episode to finish early if allowed
        if self.early_terminations:
            done = done or self._check_terminated()

        # Add termination criteria
        if done and self.print_results:
            print("Max steps per episode reached")

        return reward, done, info

    def _pre_action(self, action, policy_step=False):
        """
        Overrides the superclass method to control the robot(s) within this environment using their respective
        controllers using the passed actions and gripper control.

        Args:
            action (np.array): The control to apply to the robot(s). Note that this should be a flat 1D array that
                encompasses all actions to be distributed to each robot if there are multiple. For each section of the
                action space assigned to a single robot, the first @self.robots[i].controller.control_dim dimensions
                should be the desired controller actions and if the robot has a gripper, the next
                @self.robots[i].gripper.dof dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """
        # Single robot
        robot = self.robots[0]

        # Verify that the action is the correct dimension
        assert len(action) == robot.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            robot.action_dim, len(action)
        )

        robot.control(action, policy_step=policy_step)

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Task space limit reached
            - Task completion (tracking completed)

        Returns:
            bool: True if episode is terminated
        """
        terminated = False

        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[0].robot_model):
            if self.print_results:
                print(20 * "-" + " COLLIDED " + 20 * "-")
            terminated = True

        # Prematurely terminate if the end effector leave the play area
        if not self._check_task_space_limits():
            if self.print_results:
                print(20 * "-" + " TASK SPACE LIMIT REACHED " + 20 * "-")
            terminated = True

        # Prematurely terminate if task is completed
        if self._check_success():
            if self.print_results:
                print(20 * "-" + " TRACKING COMPLETED " + 20 * "-")
            terminated = True

        return terminated

    def _check_success(self):
        """
            Check if trajectory tracking is completed

        Returns:
            bool: True completed task
        """

        return self.current_waypoint_index +2 == self.trajectory_len

    def _check_task_space_limits(self):
        """
        Check if the eef is not too far away from mortar, works because mortar space center is at [0,0,0], does it need generalization?

        Returns:
            bool: True within task box space limits
        """

        ee_pos = self.robots[0].recent_ee_pose.current[:3]
        return not np.any(np.abs(ee_pos) > self.task_box)

    def _check_force_torque_limits(self):
        """
        Check that the robot is not exerting too much force/torque

        Returns:
            bool: True within force/torque limits
        """
        abs_ft = np.abs(self.ee_wrench)
        return not np.any(abs_ft > self.force_torque_limits)

    @property
    def action_spec(self):
        """
        Action space (low, high) for this environment

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        low, high = [], []

        for robot in self.robots:
            lo, hi = robot.action_limits
            low, high = lo[self.action_indices], hi[self.action_indices]
        return low, high

    @property
    def action_dim(self):
        """
        Size of the action space

        Returns:
            int: Action space dimension
        """
        return len(self.action_indices)

    @property
    def ee_wrench(self):
        return self.robots[0].controller.current_wrench

    def __log_details__(self, action, residual_action):
        if self.log_details:
            # save variables during training
            self.log_dict['details']['timesteps'].append(self.timestep)
            self.log_dict['details']['waypoint'].append(self.current_waypoint_index)
            self.log_dict['details']['action_in'].append(action)
            self.log_dict['details']['res_action'].append(residual_action)
            self.log_dict['details']['current_ref'].append(self.reference_trajectory[self.current_waypoint_index])
            self.log_dict['details']['current_pos'].append(self._eef_xpos)

            # just for plotting, make quat affine
            curr_quat = self._eef_xquat
            if np.dot(curr_quat,  self.prev_quat) < 0:  # if pointing in opposite directions
                curr_quat = -curr_quat
            self.prev_quat = curr_quat

            self.log_dict['details']['current_quat'].append(curr_quat)

            self.log_dict['details']['current_force_ref'].append(self.ft_action)
            self.log_dict['details']['current_force'].append(self.ee_wrench)

            # TODO separate case hand from base
            self.log_dict['details']['current_force_ref_eef_frame'].append(self.reference_force[self.current_waypoint_index])
            self.log_dict['details']['current_force_eef_frame'].append(self.robots[0].controller.eef_wrench_buf.average)

            # controller params
            self.log_dict['details']['kp'].append(self.robots[0].controller.kp.copy())

            if self.evaluate:
                self.log_filename = self.log_dir + "/step_actions_eval.npz"
                self._save_details()
            else:
                self.log_filename = self.log_dir + "/step_actions.npz"

    def _save_details(self):
        np.savez(
            self.log_filename,
            timesteps=self.log_dict['details']['timesteps'],
            waypoint=self.log_dict['details']['waypoint'],
            action_in=self.log_dict['details']['action_in'],
            res_action=self.log_dict['details']['res_action'],
            scl_action=self.log_dict['details']['scl_action'],
            crnt_ref=self.log_dict['details']['current_ref'],
            crnt_pos=self.log_dict['details']['current_pos'],
            crnt_quat=self.log_dict['details']['current_quat'],
            crnt_f_ref=self.log_dict['details']['current_force_ref'],
            crnt_f_ref_eef=self.log_dict['details']['current_force_ref_eef_frame'],
            crnt_f_eef=self.log_dict['details']['current_force_eef_frame'],
            crnt_f=self.log_dict['details']['current_force'],
            contr_kp=self.log_dict['details']['kp'],
            f_rew=self.log_dict['rewards']['force_error'],
            p_rew=self.log_dict['rewards']['traj_error']
        )
