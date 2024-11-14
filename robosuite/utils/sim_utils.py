"""
Collection of useful simulation utilities
"""

import numpy as np
from robosuite.models.base import MujocoModel


def check_contact(sim, geoms_1, geoms_2=None):
    """
    Finds contact between two geom groups.
    Args:
        sim (MjSim): Current simulation object
        geoms_1 (str or list of str or MujocoModel): an individual geom name or list of geom names or a model. If
            a MujocoModel is specified, the geoms checked will be its contact_geoms
        geoms_2 (str or list of str or MujocoModel or None): another individual geom name or list of geom names.
            If a MujocoModel is specified, the geoms checked will be its contact_geoms. If None, will check
            any collision with @geoms_1 to any other geom in the environment
    Returns:
        bool: True if any geom in @geoms_1 is in contact with any geom in @geoms_2.
    """
    # Check if either geoms_1 or geoms_2 is a string, convert to list if so
    if type(geoms_1) is str:
        geoms_1 = [geoms_1]
    elif isinstance(geoms_1, MujocoModel):
        geoms_1 = geoms_1.contact_geoms
    if type(geoms_2) is str:
        geoms_2 = [geoms_2]
    elif isinstance(geoms_2, MujocoModel):
        geoms_2 = geoms_2.contact_geoms
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        # check contact geom in geoms
        c1_in_g1 = sim.model.geom_id2name(contact.geom1) in geoms_1
        c2_in_g2 = sim.model.geom_id2name(contact.geom2) in geoms_2 if geoms_2 is not None else True
        # check contact geom in geoms (flipped)
        c2_in_g1 = sim.model.geom_id2name(contact.geom2) in geoms_1
        c1_in_g2 = sim.model.geom_id2name(contact.geom1) in geoms_2 if geoms_2 is not None else True
        if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
            return True
    return False


def get_contacts(sim, model):
    """
    Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
    geom names currently in contact with that model (excluding the geoms that are part of the model itself).
    Args:
        sim (MjSim): Current simulation model
        model (MujocoModel): Model to check contacts for.
    Returns:
        set: Unique geoms that are actively in contact with this model.
    Raises:
        AssertionError: [Invalid input type]
    """
    # Make sure model is MujocoModel type
    assert isinstance(model, MujocoModel), "Inputted model must be of type MujocoModel; got type {} instead!".format(
        type(model)
    )
    contact_set = set()
    for contact in sim.data.contact[: sim.data.ncon]:
        # check contact geom in geoms; add to contact set if match is found
        g1, g2 = sim.model.geom_id2name(contact.geom1), sim.model.geom_id2name(contact.geom2)
        if g1 in model.contact_geoms and g2 not in model.contact_geoms:
            contact_set.add(g2)
        elif g2 in model.contact_geoms and g1 not in model.contact_geoms:
            contact_set.add(g1)
    return contact_set


def compensate_ft_reading(force_reading, torque_reading, mass, com, world_rot, gravity):
    """
    Compensate for the gripper's payload in FT sensor readings by adding the gravitational effects.
    This assumes the force/torque readings are in the sensor frame and are already negative when 
    experiencing forces/torques in the positive axis directions.

    Args:
        force_reading (ndarray): Current force reading from sensor [fx, fy, fz] in sensor frame
        torque_reading (ndarray): Current torque reading from sensor [tx, ty, tz] in sensor frame
        mass (float): Mass of the payload (gripper) in kg
        com (ndarray): Center of mass of the payload [x, y, z] in sensor frame
        world_rot (ndarray): 3x3 rotation matrix from world to sensor frame
        gravity (ndarray): Gravity vector [gx, gy, gz] in world frame (typically [0, 0, -9.81])

    Returns:
        ndarray: Concatenated compensated force and torque [fx, fy, fz, tx, ty, tz] in sensor frame
                Where positive values indicate forces/torques in the positive axis directions

    Note:
        - The compensation ADDS the gravitational effects because sensor readings are typically
          negative when experiencing forces in the positive direction
        - world_rot.T @ force transforms the force from world frame to sensor frame
        - The cross product (com × force) gives the torque caused by the offset COM
    """

    # Calculate gravitational force in world frame
    force_gravity_world = mass * gravity

    # Transform gravitational force to sensor frame
    # Note: The rotation matrix transforms from world to body frame
    force_gravity_sensor = world_rot.T @ force_gravity_world

    # Calculate torque due to gravity
    # Cross product of COM vector and gravitational force in sensor frame
    torque_gravity_sensor = np.cross(com, force_gravity_sensor)

    # Compensate the readings
    compensated_force = force_reading + force_gravity_sensor
    compensated_torque = torque_reading + torque_gravity_sensor

    return np.concatenate([compensated_force, compensated_torque])
