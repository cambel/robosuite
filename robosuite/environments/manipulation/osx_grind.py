import multiprocessing
import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MortarObject, MortarVisualObject
from robosuite.models.tasks import ManipulationTask, task
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import axisangle2quat, convert_quat, mat2quat
from ur_control import spalg
import rospkg
import json
import copy

# Default Grind environment configuration
DEFAULT_GRIND_CONFIG = {
    # settings for reward
    "task_complete_reward": 50.0,  # reward per task done
    "grind_follow_reward": 0.0025830969585508424,  # reward for following the trajectory reference
    "grind_push_reward": 0.25830969585508424,  # reward for pushing into the mortar according to te force reference
    "quickness_reward": 0,  # reward for increased velocity
    "excess_accel_penalty": 0,  # penalty for end-effector accelerations over threshold
    "excess_force_penalty": 0,  # penalty for each step that the force is over the safety threshold
    "exit_task_space_penalty": 0,  # penalty for moving too far away from the mortar task space
    "bad_behavior_penalty": -100.0,  # the penalty received in case of early termination due to bad behavior
                                     # meaning one of the conditions from @termination_flag, collisions or joint limits
    "force_follow_normalization": 50.0,  # max load (N)
    "traj_follow_normalization": [0.027, 0.027, 0.085, 1, 1, 1],  # traj max? penalty?

    # settings for thresholds and flags
    "pressure_threshold_max": 20.0,  # maximum eef force allowed (N)
    "acceleration_threshold_max": 0.1,  # maximum eef acceleration allowed (ms^-2)
    "mortar_space_threshold_max": 0.1,  # maximum distance from the mortar the eef is allowed to diverge (m)
    "termination_flag":  [False, False, False],  # list of bool values representing which of the following
    # conditions are taken into account for termination of episode:
    # eef accelerations too big, eef forces too big, eef fly away;
    # collisions, joint limits and successful termination are always True

    # misc settings
    "mortar_height": 0.047,  # (m)
    "mortar_max_radius": 0.04,  # (m)
    "print_results": False,  # Whether to print results or not
    "print_early_termination": False,  # Whether to print early termination messages or not
    "get_info": False,  # Whether to grab info after each env step if not
    "use_robot_obs": True,  # if we use robot observations (proprioception) as input to the policy
    "early_terminations": True,  # Whether we allow for early terminations or not
}


TRACKING_METHOD = [
    'per_step',  # update waypoints after every step
    'per_error_threshold'  # update waypoints after the tracking error is smaller than `tracking_threshold`
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
        reward_shaping=False,
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
        tracking_threshold=0.001,
        tracking_method='per_step',
        log_dir="",
        evaluate=False,
        log_details=False,
        action_indices=range(0, 6)
    ):

        # Assert that the gripper type is Grinder
        assert (
            gripper_types == "Grinder"
        ), "Tried to specify gripper other than Grinder in Grind environment!"

        # self.log_details = log_details
        self.horizon = horizon

        self.task_config = task_config

        self.force_follow_normalization = self.task_config["force_follow_normalization"]
        self.traj_follow_normalization = np.array(self.task_config["traj_follow_normalization"])

        self.mortar_height = self.task_config["mortar_height"]
        self.mortar_radius = self.task_config["mortar_max_radius"]
        self.mortar_space_threshold_max = self.task_config["mortar_space_threshold_max"]

        # settings for table top and task space
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.task_box = np.array([self.mortar_radius, self.mortar_radius, self.mortar_height+self.table_offset[2]]) + self.mortar_space_threshold_max

        # references to follow
        self.current_waypoint_index = 0
        # Add an extra waypoint to make sure that every waypoint is
        # tracked before considering the tracking completed
        reference_trajectory = np.append(reference_trajectory, [reference_trajectory[-1]], axis=0)
        self.reference_trajectory = reference_trajectory
        self.trajectory_len = len(self.reference_trajectory)
        self.reference_force = reference_force
        self.tracking_threshold = tracking_threshold
        self.tracking_method = tracking_method

        # object placement initializer
        self.placement_initializer = placement_initializer

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

        self.action_indices = [item for item in action_indices]
        self.full_action_indices = self.action_indices

    def step(self, action):
        assert len(action) == len(self.action_indices), f"Size of action {len(action)} does not match expected {len(self.action_indices)}"

        pos_rot_action = np.zeros(6)
        pos_rot_action[self.action_indices] = action

        # print("step", action)
        # print("ee_ft", self.robots[0].controller.current_wrench[:3])
        ft_action = [0, 0, 0, 0, 0, 0]

        ctr_action = np.concatenate([pos_rot_action, ft_action])

        # online tracking of the reference trajectory
        residual_action = self._compute_relative_distance()
        factor = 1 - np.tanh(100 * self.tracking_error)
        scale_factor = np.interp(factor, [0.0, 1.0], [1, 50])
        print("error", self.tracking_error, factor, scale_factor)
        ctr_action[:6] += residual_action * scale_factor

        # print("step", ctr_action)
        return super().step(ctr_action)

    def _compute_relative_distance(self):
        relative_distance = np.zeros(6)
        if self.reference_trajectory is not None:
            relative_distance[:3] = self.reference_trajectory[self.current_waypoint_index][:3] - self._eef_xpos
            ref_quat = self.reference_trajectory[self.current_waypoint_index][3:]
            relative_distance[3:] = spalg.quaternions_orientation_error(ref_quat, self._eef_xquat)

        self.tracking_error = np.linalg.norm(relative_distance[:3])
        return relative_distance

    def _compute_relative_wrenches(self):
        relative_wrench = np.zeros(6)
        if self.reference_force is not None:
            current_waypoint = self.timestep % self.trajectory_len
            # normalized by max load; ee_ft from controller already filtered
            # relative_wrench = (self.ref_force[:, current_waypoint] - self.robots[0].controller.ee_ft.current)/self.force_follow_normalization
            relative_wrench = np.zeros(6)
            return relative_wrench
        else:
            return relative_wrench

    def reward(self, action=None):
        return 0

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

        self.visual_mortar = MortarVisualObject(
            name="mortar_visual",
        )

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
            mujoco_objects=self.mortar,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.mortar_body_id = self.sim.model.body_name2id(self.mortar.root_body)

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

        # needed in the list of observables
        @sensor(modality=f"{pf}proprio")
        def robot0_eef_force(obs_cache):
            sensor_idx = np.sum(self.sim.model.sensor_dim[: self.sim.model.sensor_name2id("gripper0_force_ee")])
            sensor_dim = self.sim.model.sensor_dim[self.sim.model.sensor_name2id("gripper0_force_ee")]
            return np.array(self.sim.data.sensordata[sensor_idx: sensor_idx + sensor_dim])

        @sensor(modality=f"{pf}proprio")
        def robot0_eef_torque(obs_cache):
            sensor_idx = np.sum(self.sim.model.sensor_dim[: self.sim.model.sensor_name2id("gripper0_torque_ee")])
            sensor_dim = self.sim.model.sensor_dim[self.sim.model.sensor_name2id("gripper0_torque_ee")]
            return np.array(self.sim.data.sensordata[sensor_idx: sensor_idx + sensor_dim])

        # sensors = [robot0_eef_force, robot0_eef_torque, robot0_relative_pose, robot0_relative_wrench]
        sensors = [robot0_relative_pose, robot0_relative_wrench]
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
        # if not self.deterministic_reset:

        #     # Sample from the placement initializer for all objects
        #     object_placements = self.placement_initializer.sample()

        #     # Loop through all objects and reset their positions
        #     for obj_pos, obj_quat, obj in object_placements.values():
        #         self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        self.current_waypoint_index = 0

        # self.collisions = 0
        # self.timestep = 0
        # self.f_excess = 0
        # self.a_excess = 0
        # self.task_space_exits = 0

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

        # Add termination criteria at the end of the trajectory tracking
        self.done = self._check_success() or done

        if self.current_waypoint_index < self.trajectory_len - 1:
            if self.tracking_method == 'per_step':
                self.current_waypoint_index += 1

            elif self.tracking_method == 'per_tracking_error':
                if self.tracking_error < self.tracking_threshold:
                    self.current_waypoint_index += 1

        return reward, self.done, info

    def _check_success(self):
        """
            Check if trajectory tracking is completed

        Returns:
            bool: True completed task
        """

        return self.current_waypoint_index == self.trajectory_len - 1

    def _check_task_space_limits(self):
        """
        Check if the eef is not too far away from mortar, works because mortar space center is at [0,0,0], does it need generalization?

        Returns:
            bool: True within task box space limits
        """

        ee_pos = self.robots[0].recent_ee_pose.current[:3]
        truth = np.abs(ee_pos) < self.task_box

        return all(truth)

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
            low, high = np.concatenate([low, lo[self.full_action_indices]]), np.concatenate([high, hi[self.full_action_indices]])
        return low, high

    @property
    def action_dim(self):
        """
        Size of the action space

        Returns:
            int: Action space dimension
        """
        return len(self.action_indices)

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
