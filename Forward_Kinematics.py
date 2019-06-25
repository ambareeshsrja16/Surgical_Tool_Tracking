import logging
import numpy as np


class SerialLinkRobot:
    def __init__(self, dh_params=None, convention="modified", logger_level=20):
        self.logger = logging.getLogger()
        self.logger.setLevel(logger_level)
        self.convention = convention

        if not dh_params:
            l_RCC = 0.4318*1e3 # millimeters
            l_tool = 0.4162*1e3
            l_P2Yaw = 0.0091*1e3
            # l_Y2Ctrl = 0.0102*1e3 End effector
            self.dh_params = np.array([[0,      np.pi/2,    0,        np.pi/2],
                                       [0,      -np.pi/2,   0,       -np.pi/2],
                                       [0,      np.pi/2,    -l_RCC,         0],
                                       [0,      0,          l_tool,         0],
                                       [0,      -np.pi/2,   0,       -np.pi/2],
                                       [l_P2Yaw, -np.pi/2,  0,       -np.pi/2]
                                       ])
            logging.info(f"Initializing Da Vinci with Modified DH parameters\n {self.dh_params}\n")
        else:
            assert isinstance(dh_params, np.ndarray)
            assert dh_params.shape[1] == 4

            self.dh_params = dh_params
            logging.info("Initializing Robot with Modified DH parameters\n")
            logging.info("f{self.dh_params}")

        self.joints = len(self.dh_params)

    def __repr__(self):
        return f"Serial Robot initialized with DH parameters{self.dh_params} in {self.convention}"

    def fkin(self, joint_params, init_pose=np.eye(4)):
        assert type(joint_params) in (np.ndarray, list)
        assert len(joint_params) == self.joints

        pose = init_pose

        # Adapt joint angles into the modified dh_params # Convert degrees to radians
        self.dh_params[0][3] += np.radians(joint_params[0])
        self.dh_params[1][3] += np.radians(joint_params[1])
        self.dh_params[2][2] += joint_params[2]   # Prismatic joint - millimeters
        self.dh_params[3][3] += np.radians(joint_params[3])
        self.dh_params[4][3] += np.radians(joint_params[4])
        self.dh_params[5][3] += np.radians(joint_params[5])

        logging.debug(f'DEBUG \n {self.dh_params}\n')

        for (a, alpha, d, theta) in self.dh_params:
            transform = self.j_1_T_j_joint_transform(a, alpha, d, theta)
            pose = pose@transform

        return pose

    def j_1_T_j_joint_transform(self, a, alpha, d, theta):
        T_X = np.array([[1, 0,               0,                 a],
                        [0, np.cos(alpha),   -np.sin(alpha),    0],
                        [0, np.sin(alpha),   np.cos(alpha),     0],
                        [0, 0,               0,                 1]])
        T_Z = np.array([[np.cos(theta), -np.sin(theta), 0,  0],
                        [np.sin(theta), np.cos(theta),  0,  0],
                        [0,             0,              1,  d],
                        [0,             0,              0,  1]])

        return T_X@T_Z


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    DaVinci = SerialLinkRobot(logger_level=20)
    joint_params = [24.397, -5.057, 188.220, 165.858, 13.575, -32.34]
    # joint_params = [0]*6
    final_pose = DaVinci.fkin(joint_params)
    print(final_pose)
