import numpy as np
from scipy.spatial.transform import Rotation as R
from engine.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit colmap)

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

        self.fovy = fovy

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[
            :3, 0
        ]  # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])
