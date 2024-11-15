import abc

import numpy as np

import robosuite.utils.transform_utils as T


# Classes for trajectory interpolation
class Interpolator(object, metaclass=abc.ABCMeta):
    """
    General interpolator interface.
    """

    @abc.abstractmethod
    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.

        Returns:
            np.array: Next interpolated step
        """
        raise NotImplementedError


class LinearInterpolator(Interpolator):
    """
    Simple class for implementing a linear interpolator.

    Abstracted to interpolate n-dimensions

    Args:
        ndim (int): Number of dimensions to interpolate

        controller_freq (float): Frequency (Hz) of the controller

        policy_freq (float): Frequency (Hz) of the policy model

        ramp_ratio (float): Percentage of interpolation timesteps across which we will interpolate to a goal position.

            :Note: Num total interpolation steps will be equal to np.floor(ramp_ratio * controller_freq / policy_freq)
                    i.e.: how many controller steps we get per action space update

        ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
            Specified string determines assumed type of input:

                `'euler'`: Euler orientation inputs
                `'quat'`: Quaternion inputs
    """

    def __init__(
        self,
        ndim,
        controller_freq,
        policy_freq,
        ramp_ratio=0.2,
        use_delta_goal=False,
        ori_interpolate=None,
    ):
        self.dim = ndim  # Number of dimensions to interpolate
        self.ori_interpolate = ori_interpolate  # Whether this is interpolating orientation or not
        self.order = 1  # Order of the interpolator (1 = linear)
        self.step = 0  # Current step of the interpolator
        self.total_steps = np.ceil(
            ramp_ratio * controller_freq / policy_freq
        )  # Total num steps per interpolator action
        self.use_delta_goal = use_delta_goal  # Whether to use delta or absolute goals (currently
        # not implemented yet- TODO)
        self.set_states(dim=ndim, ori=ori_interpolate)

    def set_states(self, dim=None, ori=None):
        """
        Updates self.dim and self.ori_interpolate.

        Initializes self.start and self.goal with correct dimensions.

        Args:
            ndim (None or int): Number of dimensions to interpolate

            ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
                Specified string determines assumed type of input:

                    `'euler'`: Euler orientation inputs
                    `'quat'`: Quaternion inputs
        """
        # Update self.dim and self.ori_interpolate
        self.dim = dim if dim is not None else self.dim
        self.ori_interpolate = ori if ori is not None else self.ori_interpolate

        # Set start and goal states
        if self.ori_interpolate is not None:
            if self.ori_interpolate == "euler":
                self.start = np.zeros(3)
            else:  # quaternions
                self.start = np.array((0, 0, 0, 1))
        else:
            self.start = np.zeros(self.dim)
        self.goal = np.array(self.start)

    def set_goal(self, goal):
        """
        Takes a requested (absolute) goal and updates internal parameters for next interpolation step

        Args:
            np.array: Requested goal (absolute value). Should be same dimension as self.dim
        """
        # First, check to make sure requested goal shape is the same as self.dim
        if goal.shape[0] != self.dim:
            print("Requested goal: {}".format(goal))
            raise ValueError(
                "LinearInterpolator: Input size wrong for goal; got {}, needs to be {}!".format(goal.shape[0], self.dim)
            )

        # Update start and goal
        self.start = np.array(self.goal)
        self.goal = np.array(goal)

        # Reset interpolation steps
        self.step = 0

    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.

        NOTE: If this interpolator is for orientation, it is assumed to be receiving either euler angles or quaternions

        Returns:
            np.array: Next position in the interpolated trajectory
        """
        # Grab start position
        x = np.array(self.start)
        # Calculate the desired next step based on remaining interpolation steps
        if self.ori_interpolate is not None:
            # This is an orientation interpolation, so we interpolate linearly around a sphere instead
            goal = np.array(self.goal)
            if self.ori_interpolate == "euler":
                # this is assumed to be euler angles (x,y,z), so we need to first map to quat
                x = T.mat2quat(T.euler2mat(x))
                goal = T.mat2quat(T.euler2mat(self.goal))

            # Interpolate to the next sequence
            x_current = T.quat_slerp(x, goal, fraction=(self.step + 1) / self.total_steps)
            if self.ori_interpolate == "euler":
                # Map back to euler
                x_current = T.mat2euler(T.quat2mat(x_current))
        else:
            # This is a normal interpolation
            dx = (self.goal - x) / (self.total_steps - self.step)
            x_current = x + dx

        # Increment step if there's still steps remaining based on ramp ratio
        if self.step < self.total_steps - 1:
            self.step += 1

        # Return the new interpolated step
        return x_current


def generate_mortar_trajectory(mortar_diameter, desired_height, n_steps, default_quat=np.array([0, -1, 0, 0]), fraction=None):
    """
    Generate a trajectory to trace the surface of an upward-facing bowl at a given height.
    The pen orientation at the center (0,0,0) is represented by quaternion [0,-1,0,0].

    Args:
        mortar_diameter (float): Diameter of the bowl in meters
        desired_height (float): Desired height from the bottom of the bowl in meters
        n_steps (int): Number of points in the trajectory
        default_quat (list): Quaternion [qx, qy, qz, qw] representing orientation at the bowl center at (0,0,0)
        fraction: (float): If defined, the final quaternion returned is the slerp fraction from the default_quat to the 
                           corresponding normal vector

    Returns:
        np.array: Array of shape (n_steps, 7) containing [x, y, z, qx, qy, qz, qw]
                 for each point in the trajectory
    """
    # Step 1: Calculate bowl parameters
    radius = mortar_diameter / 2

    # Step 2: Verify if desired height is valid
    if desired_height > radius:
        raise ValueError("Desired height cannot be greater than bowl radius")

    # Step 3: Calculate radius of the circle at desired height
    # For upward facing bowl: r^2 = R^2 - (R-h)^2
    circle_radius = np.sqrt(radius**2 - (radius - desired_height)**2)

    # Step 4: Generate points along a circle at the desired height
    theta = np.linspace(0, 2*np.pi, n_steps)
    x = circle_radius * np.cos(theta)
    y = circle_radius * np.sin(theta)
    z = np.full_like(theta, desired_height)

    # Step 5: Calculate normal vectors at each point
    # For an upward facing bowl, the normal vector points outward from the center of curvature
    normals = np.zeros((n_steps, 3))
    for i in range(n_steps):
        point = np.array([x[i], y[i], z[i] - radius])  # Shift center of curvature to (0,0,-R)
        normal = -point / np.linalg.norm(point)  # Normalize and negate for outward normal
        normals[i] = normal

    # Step 6: Convert normal vectors to quaternions considering initial orientation
    quaternions = np.zeros((n_steps, 4))

    initial_rotation = T.quat2mat(default_quat)

    for i in range(n_steps):
        normal = normals[i]

        # Calculate rotation from [0, 0, 1] to normal vector
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, normal)
        s = np.linalg.norm(v)

        if s < 1e-10:  # If vectors are parallel
            if normal[2] > 0:  # Same direction
                surface_rotation = T.quat2mat([0, 0, 0, 1])
            else:  # Opposite direction
                surface_rotation = T.quat2mat([1, 0, 0, 0])
        else:
            c = np.dot(z_axis, normal)
            v_skew = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
            R_matrix = np.eye(3) + v_skew + np.matmul(v_skew, v_skew) * (1 - c) / (s * s)
            surface_rotation = R_matrix

        # Compose rotations: first apply initial orientation, then surface normal rotation
        final_rotation = surface_rotation @ initial_rotation
        final_quaternion = T.mat2quat(final_rotation)
        if fraction:
            final_quaternion = T.quat_slerp(default_quat, final_quaternion, fraction=fraction)
        quaternions[i] = final_quaternion

    # Step 7: Combine positions and orientations
    trajectory = np.column_stack((x, y, z, quaternions))
    trajectory = np.concatenate([trajectory, [trajectory[0]]])

    return trajectory
