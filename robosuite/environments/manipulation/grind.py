import multiprocessing
import numpy as np
import time

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MortarObject, MortarVisualObject
from robosuite.models.tasks import ManipulationTask, task
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


# Default Grind environment configuration TODO settle these
DEFAULT_GRIND_CONFIG = {
    # settings for reward
    "task_complete_reward": 5.0,  # reward per episode done
    "arm_limit_collision_penalty": -10.0,  # penalty for reaching joint limit or arm collision with the table
    "grind_contact_reward": 0.01,  # reward for contacting something with the grinder tool
    "ee_accel_penalty": 0,  # penalty for large end-effector accelerations
    "excess_force_penalty_mul": 0.05,  # penalty for each step that the force is over the safety threshold
    # settings for thresholds
    "contact_threshold": 0.5,  # Minimum eef force to qualify as contact [N]
    "pressure_threshold_max": 20.0,  # maximum force allowed (N)
    "mortar_space_threshold": 0.1,  # maximum distance from the mortar the eef is allowed to diverge (m)
    # misc settings
    "mortar_height": 0.0047,
    "mortar_max_radius": 0.004,
    "print_results": True,  # Whether to print results or not
    "get_info": False,  # Whether to grab info after each env step if not
    "use_robot_obs": True,  # if we use robot observations (proprioception) as input to the policy
    "use_contact_obs": True,  # if we use a binary observation for whether robot is in contact or not
    "early_terminations": True,  # Whether we allow for early terminations or not
    "use_condensed_obj_obs": True,  # Whether to use condensed object observation representation (only applicable if obj obs is active)
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
    ):

        # Assert that the gripper type is None
        assert (
            gripper_types == "Grinder"
        ), "Tried to specify gripper other than Grinder in Grind environment!"

        # Get config
        self.task_config  = task_config if task_config is not None else DEFAULT_GRIND_CONFIG

        self.horizon = horizon
        # settings for the reward
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.arm_limit_collision_penalty = self.task_config["arm_limit_collision_penalty"]
        self.grind_contact_reward = self.task_config["grind_contact_reward"]
        self.ee_accel_penalty = self.task_config["ee_accel_penalty"]
        self.excess_force_penalty_mul = self.task_config["excess_force_penalty_mul"]
        # Final reward computation
        self.task_complete_reward = self.task_config["task_complete_reward"]

        # settings for thresholds
        self.contact_threshold = self.task_config["contact_threshold"]
        self.pressure_threshold_max = self.task_config["pressure_threshold_max"]
        self.mortar_space_threshold = self.task_config["mortar_space_threshold"]

        # misc settings
        self.print_results = self.task_config["print_results"]
        self.get_info = self.task_config["get_info"]
        self.use_robot_obs = self.task_config["use_robot_obs"]
        self.use_contact_obs = self.task_config["use_contact_obs"]
        self.early_terminations = self.task_config["early_terminations"]
        self.use_condensed_obj_obs = self.task_config["use_condensed_obj_obs"]
        self.mortar_height = self.task_config["mortar_height"]
        self.mortar_radius = self.task_config["mortar_max_radius"]

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # set other attributes
        self.collisions = 0

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

    def reward(self, action=None): #TODO
        """
        Reward function for the task.
        Sparse un-normalized reward:

            - a discrete reward of self.task_complete_reward is provided if grinding has been done for a predetermined amount of steps

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the mortar

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # Neg Reward from unwanted behaviours TODO properly deffine the if branches
        if self.check_contact(self.robots[0].robot_model):
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        elif self.robots[0].check_q_limits():
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        elif not self._check_task_space_limits():
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        elif self._surpassed_forces():
            if self.reward_shaping:
                reward = self.arm_limit_collision_penalty
            self.collisions += 1
        else:
            # If the arm is not colliding, not in joint limits, or flying away,
            # check if we are grinding
            # (we don't want to reward grinding if there are unsafe situations)

            # sparse completion reward
            if self._check_success():
                reward = 2.25

            # use a shaping reward
            elif self.reward_shaping:

                # reaching reward
                mortar_pos = self.sim.data.body_xpos[self.mortar_body_id]
                gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
                dist = np.linalg.norm(gripper_site_pos - mortar_pos)
                reaching_reward = 1 - np.tanh(10.0 * dist)
                reward += reaching_reward


        # Printing results
        if self.print_results:
            string_to_print = (
                "Process {pid}, timestep {ts:>4}: reward: {rw:8.4f}, collisions: {sc:>3}".format(
                    pid=id(multiprocessing.current_process()),
                    ts=self.timestep,
                    rw=reward,
                    sc=self.collisions,
                )
            )
            print(string_to_print)

        # Scale reward if requested #TODO
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

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
            return np.array(self.sim.data.sensordata[sensor_idx : sensor_idx + sensor_dim])

        @sensor(modality=f"{pf}proprio")
        def robot0_eef_torque(obs_cache):
            sensor_idx = np.sum(self.sim.model.sensor_dim[: self.sim.model.sensor_name2id("gripper0_torque_ee")])
            sensor_dim = self.sim.model.sensor_dim[self.sim.model.sensor_name2id("gripper0_torque_ee")]
            return np.array(self.sim.data.sensordata[sensor_idx : sensor_idx + sensor_dim])


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

        if self.get_info: # TODO what else to add here
            info["add_vals"] = ["colls"]
            info["colls"] = self.collisions

        # allow episode to finish early if allowed
        if self.early_terminations:
            done = done or self._check_terminated()

        return reward, done, info
    # def _pre_action(self, action): TODO needed?

    def _check_success(self):
        """
        Check if task succeeded (finished working for predetermined amount of  timesteps)

        Returns:
            bool: True completed task
        """

        return  self.timestep > self.horizon # TODO left like this?, have another timesteps/time variable?

    def _check_task_space_limits(self): # returns True if it is within limits
        """
        Check if the eef is not too far away from mortar

        Returns:
            bool: True within task box space limits
        """

        ee_pos = self.robots[0].recent_ee_pose.current[:3]
        task_box = np.array([self.mortar_radius, self.mortar_radius, self.mortar_height+self.table_offset[2]]) + self.mortar_space_threshold
        truth = np.abs(ee_pos) < task_box
        print(all(truth))
        print(truth)

        return all(truth)

    def _surpassed_forces(self): # returns True if forces surpassed predetermined threshold
        dft = self.robots[0].controller.ee_ft.current

        return not all(i < self.pressure_threshold_max for i in dft)

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:

            - Collision of the robot with table
            - Task completion (amount of @horizon timesteps passed)
            - Joint Limit reached
            - Fly off mortar neighboring space
            - Force way too big

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if contacting the table with the arm
        if self.check_contact(self.robots[0].robot_model): # TODO check if just arm or eef as well
            if self.print_results:
                print(40 * "-" + " COLLIDED " + 40 * "-")
            terminated = True

        # Prematurely terminate if task is success
        if self._check_success():
            if self.print_results:
                print(40 * "+" + " FINISHED GRINDING " + 40 * "+")
            terminated = True

        # Prematurely terminate if joint limits reached
        if self.robots[0].check_q_limits():
            if self.print_results:
                print(40 * "-" + " JOINT LIMIT " + 40 * "-")
            terminated = True

        # Prematurely terminate if robot starts to diverge too much from the mortar space
        if not self._check_task_space_limits():
            if self.print_results:
                print(40 * "-" + " EXIT MORTAR SPACE " + 40 * "-")
            terminated = True

        # Prematurely terminate if robot acts with a force over predetermined threshold
        if self._surpassed_forces():
            if self.print_results:
                print(40 * "-" + " FORCE LIMIT SURPASSED " + 40 * "-")
            terminated = True

        #TODO add for accelerations

        return terminated

    @property
    def _has_gripper_contact(self): # TODO - used as observable, is it needed?
        """
        Determines whether the gripper is making contact with an object, as defined by the eef force surpassing
        a certain threshold defined by self.contact_threshold

        Returns:
            bool: True if contact is surpasses given threshold magnitude
        """
        return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias) > self.contact_threshold
