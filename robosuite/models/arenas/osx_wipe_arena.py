import numpy as np
from robosuite.models.arenas import Arena  # Import the general Arena class
from robosuite.models.objects import CylinderObject
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements, xml_path_completion
from robosuite.utils.mjcf_utils import new_body, array_to_string


class OSXWipeArena(Arena):
    """
    Workspace that loads an existing 'osx_arena.xml' arena and places visual markers on its table.
    """

    def __init__(
            self,
            wiping_area=(0.8, 0.8, 0.05),
            num_markers=10,
            line_width=0.02,
            center_pose=[-0.174, -0.14],
            table_height = 0.8555 + 0.0025/2 + 0.003,
            coverage_factor=0.9,
            two_clusters=False,
            seed=0,
            xml="arenas/osx_arena.xml"
    ):
        super().__init__(xml_path_completion(xml))
        self.rng = np.random.default_rng(seed)
        self.wiping_area = np.array(wiping_area)
        self.wiping_area_half_size = self.wiping_area / 2

        self.table_height = table_height
        self.center_pos = center_pose

        self.num_markers = num_markers
        self.line_width = line_width
        self.coverage_factor = coverage_factor
        self.two_clusters = two_clusters
        self.markers = []

        # Load and configure the arena
        self.configure_location()

    def configure_location(self):
        # Assume the table object's name in osx_arena.xml is 'table'
        # Adjust 'table' to the correct name if different
        pos = self.sample_start_pos()
        table = find_elements(root=self.worldbody, tags="body", attribs={"name": "workspace"}, return_first=True)
        if not table:
            raise ValueError("Table not found in osx_arena.xml")

        # Define dirt material for markers
        dirt = CustomMaterial(texture="Dirt", tex_name="dirt", mat_name="dirt_mat")

        # Place visual markers
        for i in range(self.num_markers):
            marker_name = f"contact{i}"
            marker = CylinderObject(name=marker_name, size=[self.line_width / 2, 0.001], rgba=[1, 1, 1, 1], material=dirt, obj_type="visual", joints=None)
            self.merge_assets(marker)
            table.append(marker.get_obj())
            self.markers.append(marker)
            pos = self.sample_path_pos(pos)

    def reset_arena(self, sim):
        """
        Reset the visual marker locations in the environment. Requires @sim (MjSim) reference to be passed in so that
        the Mujoco sim can be directly modified

        Args:
            sim (MjSim): Simulation instance containing this arena and visual markers
        """
        # Sample new initial position and direction for generated marker paths
        pos = self.sample_start_pos()

        # Loop through all visual markers
        for i, marker in enumerate(self.markers):
            # If we're using two clusters, we resample the starting position and direction at the halfway point
            if self.two_clusters and i == int(np.floor(self.num_markers / 2)):
                pos = self.sample_start_pos()
            # Get IDs to the body, geom, and site of each marker
            body_id = sim.model.body_name2id(marker.root_body)
            geom_id = sim.model.geom_name2id(marker.visual_geoms[0])
            site_id = sim.model.site_name2id(marker.sites[0])
            # Determine new position for this marker
            # position = np.array([pos[0], pos[1], self.fixture_half_size[2]])
            position = np.array([pos[0]+self.center_pos[0], pos[1]+self.center_pos[1], self.table_height])
            # Set the current marker (body) to this new position
            sim.model.body_pos[body_id] = position
            # Reset the marker visualization -- setting geom rgba alpha value to 1
            sim.model.geom_rgba[geom_id][3] = 1
            # Hide the default visualization site
            sim.model.site_rgba[site_id][3] = 0
            # Sample next values in local marker trajectory
            pos = self.sample_path_pos(pos)

    def sample_start_pos(self):
        """
        Helper function to return sampled start position of a new dirt (peg) location

        Returns:
            np.array: the (x,y) value of the newly sampled dirt starting location
        """
        # First define the random direction that we will start at
        self.direction = self.rng.uniform(-np.pi, np.pi)

        return np.array(
            (
                self.rng.uniform(
                    - self.wiping_area_half_size[0] * self.coverage_factor + self.line_width / 2,
                    + self.wiping_area_half_size[0] * self.coverage_factor - self.line_width / 2,
                ),
                self.rng.uniform(
                    - self.wiping_area_half_size[1] * self.coverage_factor + self.line_width / 2,
                    + self.wiping_area_half_size[1] * self.coverage_factor - self.line_width / 2,
                ),
            )
        )

    def sample_path_pos(self, pos):
        """
        Helper function to add a sampled dirt (peg) position to a pre-existing dirt path, whose most
        recent dirt position is defined by @pos

        Args:
            pos (np.array): (x,y) value of most recent dirt position

        Returns:
            np.array: the (x,y) value of the newly sampled dirt position to add to the current dirt path
        """
        # Random chance to alter the current dirt direction
        if self.rng.uniform(0, 1) > 0.7:
            self.direction += self.rng.normal(0, 0.5)

        posnew0 = pos[0] + 0.01 * np.sin(self.direction)
        posnew1 = pos[1] + 0.01 * np.cos(self.direction)

        # We keep resampling until we get a valid new position that's on the table
        while (
            abs(posnew0) >= self.wiping_area_half_size[0] * self.coverage_factor - self.line_width / 2
            or abs(posnew1) >= self.wiping_area_half_size[1] * self.coverage_factor - self.line_width / 2
        ):
            self.direction += np.random.normal(0, 0.5)
            posnew0 = pos[0] + 0.005 * np.sin(self.direction)
            posnew1 = pos[1] + 0.005 * np.cos(self.direction)

        # Return this newly sampled position
        return np.array((posnew0, posnew1))

    # Implement reset_arena, sample_start_pos, and sample_path_pos as in the original class,
    # adjusting as necessary for the osx_arena specifics.
