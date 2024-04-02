import multiprocessing
import numpy as np
from scipy.spatial.distance import cdist

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MortarObject, MortarVisualObject
from robosuite.models.tasks import ManipulationTask, task
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import axisangle2quat, convert_quat, mat2quat
from ur_control import spalg


# Default Grind environment configuration
DEFAULT_GRIND_CONFIG = {
    # settings for reward
    "task_complete_reward": 50.0,  # reward per task done
    "grind_follow_reward": 1,  # reward for following the trajectory reference
    "grind_push_reward": 0,  # reward for pushing into the mortar according to te force reference
    "quickness_reward": 0,  # reward for increased velocity
    "excess_accel_penalty": 0,  # penalty for end-effector accelerations over threshold
    "excess_force_penalty": 0,  # penalty for each step that the force is over the safety threshold
    "exit_task_space_penalty": 0,  # penalty for moving too far away from the mortar task space
    "bad_behavior_penalty": -100.0,  # the penalty received in case of early termination due to bad behavior
                                     # meaning one of the conditions from @termination_flag, collisions or joint limits

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


class Grind(SingleArmEnv):
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
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
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
        task_config=None,
        ref_traj=None,
        ref_force=None,
        log_dir="",
        evaluate=False,
        log_details=False,
    ):

        # Assert that the gripper type is Grinder
        assert (
            gripper_types == "Grinder"
        ), "Tried to specify gripper other than Grinder in Grind environment!"

        # Get config
        self.task_config = task_config if task_config is not None else DEFAULT_GRIND_CONFIG
        self.log_details = log_details
        self.horizon = horizon
        # Final reward computation
        self.task_complete_reward = self.task_config["task_complete_reward"]
        # settings for the reward
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        # Normalization factor = theoretical best episode return
        self.reward_normalization_factor = 1.0 / self.task_complete_reward

        self.grind_follow_reward = self.task_config["grind_follow_reward"]
        self.grind_push_reward = self.task_config["grind_push_reward"]
        self.quickness_reward = self.task_config["quickness_reward"]

        self.excess_accel_penalty = self.task_config["excess_accel_penalty"]
        self.excess_force_penalty = self.task_config["excess_force_penalty"]
        self.exit_task_space_penalty = self.task_config["exit_task_space_penalty"]
        self.bad_behavior_penalty = self.task_config["bad_behavior_penalty"]

        # settings for thresholds
        self.pressure_threshold_max = self.task_config["pressure_threshold_max"]
        self.acceleration_threshold_max = self.task_config["acceleration_threshold_max"]
        self.mortar_space_threshold_max = self.task_config["mortar_space_threshold_max"]
        self.termination_flag = self.task_config["termination_flag"]

        # misc settings
        self.print_results = self.task_config["print_results"]
        self.print_early_termination = self.task_config["print_early_termination"]
        self.get_info = self.task_config["get_info"]
        self.use_robot_obs = self.task_config["use_robot_obs"]
        self.early_terminations = self.task_config["early_terminations"]
        self.mortar_height = self.task_config["mortar_height"]
        self.mortar_radius = self.task_config["mortar_max_radius"]
        self.log_dir = log_dir
        self.evaluate = evaluate

        # settings for table top and task space
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.task_box = np.array([self.mortar_radius, self.mortar_radius, self.mortar_height+self.table_offset[2]]) + self.mortar_space_threshold_max

        # references to follow
        self.ref_traj = ref_traj
        self.ref_force = ref_force

        # Assert that if both reference trajectory and reference force are given, they have the same length
        if self.ref_force is np.ndarray and self.ref_traj is np.ndarray:
            assert (
                len(self.ref_force) == len(self.ref_traj)
            ), "Please input reference trajectory and reference force with the same dimensions"

        # Assert that if at least one reference given it has enough waypoints for finishing in @horizon timesteps
        if isinstance(self.ref_force, np.ndarray) or isinstance(self.ref_traj, np.ndarray):
            try:
                self.traj_len = self.ref_force.shape[1]

            except:
                self.traj_len = self.ref_traj.shape[1]

            assert (
                self.traj_len <= self.horizon
            ), "Reference trajectory or force cannot be completed in the specified horizon"

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # set other attributes
        self.collisions = 0
        self.f_excess = 0
        self.a_excess = 0
        self.task_space_exits = 0

        # for step log in npz
        self.timesteps = []
        self.waypoint = []
        self.action_in = []
        self.res_action = []
        self.scl_action = []
        self.current_ref = []
        self.sum_action = []
        self.current_pos = []

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
        try:

            # online tracking of the reference trajectory
            residual_action = np.zeros_like(action)
            current_waypoint = self.timestep % self.traj_len
            residual_action[:3] = self.ref_traj[:3, current_waypoint] - self.robots[0].controller.ee_pos
            ee_quat = mat2quat(self.robots[0].controller.ee_ori_mat)
            ref_quat = axisangle2quat(self.ref_traj[3:, current_waypoint])
            residual_action[3:] = spalg.quaternions_orientation_error(ref_quat, ee_quat)

            # add policy action
            scaled_action = np.interp(action, [-1, 1], [-0.02, 0.02])  # kind of linear mapping to controller.json min max output

            # let z be taken only from the residual action
            scaled_action[2] = 0.0

            if self.log_details:
                # save variables during training
                self.timesteps.append(self.timestep)
                self.waypoint.append(current_waypoint)
                self.action_in.append(action)
                self.res_action.append(residual_action)
                self.scl_action.append(scaled_action)
                self.current_ref.append(self.ref_traj[:, current_waypoint])
                self.sum_action.append(residual_action + scaled_action)
                self.current_pos.append(self.robots[0].controller.ee_pos)

                if self.evaluate:
                    log_filename = self.log_dir + "/step_actions_eval.npz"
                else:
                    log_filename = self.log_dir + "/step_actions.npz"

                np.savez(
                    log_filename,
                    timesteps=self.timesteps,
                    waypoint=self.waypoint,
                    action_in=self.action_in,
                    res_action=self.res_action,
                    scl_action=self.scl_action,
                    crnt_ref=self.current_ref,
                    sum_action=self.sum_action,
                    crnt_pos=self.current_pos
                )

            action = residual_action + scaled_action
            return super().step(action)
        except:
            return super().step(action)


    def reward(self, action=None):
        """
        Reward function for the task.
        Sparse un-normalized reward:

            - a discrete reward of self.task_complete_reward is provided if grinding has been done for a predetermined amount of steps

        Un-normalized summed components if using reward shaping:

            - Following: in [-inf, 0] , proportional to distance between end-effector pose and
              the desired reference trajectory (negative)
            - Pushing: in [-inf, 0], proportional to difference between end-effector wrench and
              the desired reference force profile (negative)
            - Quickness: in [0, inf], proportional to velocity, rewarding increase in velocity (positive)
            - Collision / Joint Limit Penalty: in {self.bad_behavior_penalty, 0}, nonzero if robot arm
              is colliding with an object or reaching joint limits
            - Large Force Penalty: in [-inf, 0], in the case the episode does not stop when exiting
              (based on termination_flag):scaled by grinding force and directly proportional to
              self.excess_force_penalty whe the current force exceeds self.pressure_threshold_max
            - Large Acceleration Penalty: in [-inf, 0], in the case the episode does not stop when threshold passed
              (based on termination_flag): scaled by estimated grinder acceleration and directly proportional to
              self.excess_accel_penalty when acceleration exceeds self.acceleration_threshold_max
            - Exit Task Space Penalty: in [-inf, 0], in the case the episode does not stop when exiting 
              (based on termination_flag): scaled by distance of grinder from mortar task space and directly
              proportional to self.exit_task_space_penalty when distance exceeds self.mortar_space_threshold_max


        Note that the final reward is normalized and scaled by task_complete_reward, which is the best reward an episode can get

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # If the arm does not present unwanted behaviors (collisions, reaching joint limits,
        # or other conditions based on @termination_flag), calculate reward
        # (we don't want to reward grinding if there are unsafe situations)
        if self.check_contact(self.robots[0].robot_model):
            if self.reward_shaping:
                reward = self.bad_behavior_penalty
            self.collisions += 1
        elif self.robots[0].check_q_limits():
            if self.reward_shaping:
                reward = self.bad_behavior_penalty
            self.collisions += 1
        elif self.termination_flag[0] and self._surpassed_accel():
            if self.reward_shaping:
                reward = self.bad_behavior_penalty
            self.a_excess += 1
        elif self.termination_flag[1] and self._surpassed_forces():
            if self.reward_shaping:
                reward = self.bad_behavior_penalty
            self.f_excess += 1
        elif self.termination_flag[2] and not self._check_task_space_limits():
            if self.reward_shaping:
                reward = self.bad_behavior_penalty
            self.task_space_exits += 1
        else:

            # sparse completion reward
            if self._check_success():
                reward = self.task_complete_reward

            # use a shaping reward
            elif self.reward_shaping:
                ee_pos = self.robots[0].controller.ee_pos[:3] # in absolute, like self.ref_traj
                try:
                    current_waypoint = self.timestep % self.traj_len

                    # Reward for pushing into mortar with desired linear forces
                    if self.ref_force is not None:
                        ee_ft = self.robots[0].controller.ee_ft.current[:3]
                        distance_from_ref_force = np.linalg.norm(self.ref_force[:3, current_waypoint] - ee_ft)
                        reward -= self.grind_push_reward * distance_from_ref_force

                    # Reward for following desired linear trajectory
                    if self.ref_traj is not None:
                        distance_from_ref_traj = np.linalg.norm(self.ref_traj[:3, current_waypoint] - ee_pos)
                        reward -= self.grind_follow_reward * distance_from_ref_traj
                except:
                    pass  # situation when no ref given but why would you do that to it

                # Reward for increased linear velocity
                reward += self.quickness_reward * np.mean(abs(self.robots[0].recent_ee_vel.current[:3]))

                # Cases when threshold surpassed but we don't terminate episode because of that
                # Penalize excessive accelerations
                if self._surpassed_accel():
                    self.a_excess += 1
                    reward -= self.excess_accel_penalty * np.mean(abs(self.robots[0].recent_ee_acc.current))

                # Penalize excessive wrenches with the end-effector
                if self._surpassed_forces():
                    self.f_excess += 1
                    reward -= self.excess_force_penalty * np.mean(abs(self.robots[0].controller.ee_ft.current))

                # Penalize flying off mortar space
                if not self._check_task_space_limits():
                    self.task_space_exits += 1
                    distance_from_mortar = self.eef_dist_from_mortar(ee_pos)
                    reward -= self.exit_task_space_penalty * distance_from_mortar

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale * self.reward_normalization_factor

        # Printing results
        if self.print_results:
            string_to_print = (
                "Process {pid}, timestep {ts:>4}: reward: {rw:8.4f}, collisions: {sc:>3}, f_excess: {fe:>3}, a_excess: {ae:>3}, task_space_exit: {te:>3}".format(
                    pid=id(multiprocessing.current_process()),
                    ts=self.timestep,
                    rw=reward,
                    sc=self.collisions,
                    fe=self.f_excess,
                    ae=self.a_excess,
                    te=self.task_space_exits,
                )
            )
            print(string_to_print)

        return reward

    def eef_dist_from_mortar(self,eef_pos):
        # assumes cube is axis aligned and with corners +-self.task_box[0], +-self.task_box[1], +-self.task_box[2]
        x = eef_pos[0]
        y = eef_pos[1]
        z = eef_pos[2]

        dist = np.sqrt( (np.max([0.0, np.abs(x)-self.task_box[0]]))**2 + (np.max([0.0, np.abs(y) - self.task_box[1]]))**2 + (np.max([0.0, np.abs(z) - self.task_box[2]]))**2)
        return dist

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        self.robots[0].init_qpos = np.array([-0.242, -0.867, 1.993, -2.697, -1.571, -1.812])

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

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # mortar-related observables
            @sensor(modality=modality)
            def mortar_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.mortar_body_id])

            @sensor(modality=modality)
            def mortar_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.mortar_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_mortar_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["mortar_pos"]
                    if f"{pf}eef_pos" in obs_cache and "mortar_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [mortar_pos, mortar_quat, gripper_to_mortar_pos, robot0_eef_force, robot0_eef_torque]
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

        # ee resets - bias at initial state
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        self.collisions = 0
        self.timestep = 0
        self.f_excess = 0
        self.a_excess = 0
        self.task_space_exits = 0

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

        # Update force bias
        if np.linalg.norm(self.ee_force_bias) == 0:
            self.ee_force_bias = self.robots[0].ee_force
            self.ee_torque_bias = self.robots[0].ee_torque

        if self.get_info:
            info["add_vals"] = ["colls", "f_excess", "a_excess", "task_spae_exit"]
            info["colls"] = self.collisions
            info["f_excess"] = self.f_excess
            info["a_excess"] = self.a_excess
            info["task_space_exit"] = self.task_space_exits

        # allow episode to finish early if @self.early_termination True
        if self.early_terminations:
            done = done or self._check_terminated()

        return reward, done, info

    def _check_success(self):
        """
        Check if task succeeded (finished working for @horizon amount of timesteps)

        Returns:
            bool: True completed task
        """

        return self.timestep > self.horizon

    def _check_task_space_limits(self):
        """
        Check if the eef is not too far away from mortar, works because mortar space center is at [0,0,0], does it need generalization?

        Returns:
            bool: True within task box space limits
        """

        ee_pos = self.robots[0].recent_ee_pose.current[:3]
        truth = np.abs(ee_pos) < self.task_box

        return all(truth)

    def _surpassed_forces(self):
        """
        Check if any of the eef wrenches surpassed predetermined threshold @self.pressure_threshold_max

        Returns:
            bool: True if at least one of the eef wrenches out of bounds
        """

        dft = self.robots[0].controller.ee_ft.current

        return not all(i < self.pressure_threshold_max for i in dft)

    def _surpassed_accel(self):
        """
        Check if any of the eef accelerations surpassed predetermined threshold @self.acceleration_threshold_max

        Returns:
            bool: True if at least one of the eef acceleration out of bounds
        """

        dacc = self.robots[0].recent_ee_acc.current

        return not all(i < self.acceleration_threshold_max for i in dacc)

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Collision of the robot with table
            - Task completion (amount of @horizon timesteps passed)
            - Joint Limit reached

        The following conditions CAN lead to termination, based on @termination_flag boolean values:

            - Accelerations over a predetermined threshold
            - Forces over a predetermined threshold
            - End-effector fly off, too far from the mortar task space

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        messages = [" COLLIDED ", " FINISHED GRINDING ", " JOINT LIMIT ", " EXCESS ACCELERATION ", " EXCESS FORCES ", " EXIT MORTAR SPACE "]
        termination_conditions_to_check = [True, True, True] + self.termination_flag

        conditions = [self.check_contact(self.robots[0].robot_model),
                      self._check_success(),
                      self.robots[0].check_q_limits(),
                      self._surpassed_accel(),
                      self._surpassed_forces(),
                      not self._check_task_space_limits()]

        for i in range(0, len(conditions)):
            if [a and b for a, b in zip(termination_conditions_to_check, conditions)][i] == True:
                if self.print_early_termination:
                    print(40 * "-" + messages[i] + 40 * "-")
                terminated = True

        return terminated
