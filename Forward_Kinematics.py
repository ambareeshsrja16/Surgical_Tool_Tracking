import logging
import numpy as np
import transforms3d
from numpy.linalg import inv


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

        self.joints = len(self.dh_params)+1  # for the 7th (jaw)

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

        for index, (a, alpha, d, theta) in enumerate(self.dh_params):
            transform = self.j_1_T_j_joint_transform(a, alpha, d, theta)
            pose = pose@transform
            if index == 3:
                pose_after_joint_4 = pose
            if index == 4:
                pose_after_joint_5 = pose
            if index == 5:
                pose_after_joint_6 = pose

        jaw = np.radians(joint_params[6])  # convert to radians
        eHeL, eHeR = np.eye(4), np.eye(4)
        eHeL[1][1] = np.cos(jaw/2)
        eHeL[1][2] = -np.sin(jaw/2)
        eHeL[2][1] = np.sin(jaw/2)
        eHeL[2][2] = np.cos(jaw/2)
        eHeR[1][1] = np.cos(jaw/2)
        eHeR[1][2] = np.sin(jaw/2)
        eHeR[2][1] = -np.sin(jaw/2)
        eHeR[2][2] = np.cos(jaw/2)

        return pose, pose_after_joint_4, pose_after_joint_5, pose_after_joint_6, eHeL, eHeR

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

    @staticmethod
    def get_joint_angles_from_reading(angles):
        joint_params = np.rad2deg(angles)  # Use all 7 #Units are radians, convert to degrees
        joint_params[2] = angles[2] * 1000   # 3 parameter is in metres, convert to millimeters
        return joint_params


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # Hand-eye 1
    q_handeye1 = [-0.122, -0.316, 0.866, -0.368] # hand eye rotation in quaternions [w, x, y, z]
    R_handeye1 = transforms3d.quaternions.quat2mat(q_handeye1)
    t_handeye1 = [0.094, -0.068, -0.030] # m
    t_handeye1 = [i*1000 for i in t_handeye1]  # mm
    cTb1 = np.identity(4)
    cTb1[:-1, :-1] = R_handeye1
    cTb1[:-1, -1] = t_handeye1
    #print("cTb1\n", cTb1)

    # Hand-eye 2
    q_handeye2 = [0.095, 0.274, 0.861, -0.418]  # hand eye rotation in quaternions [w, x, y, z]
    R_handeye2 = transforms3d.quaternions.quat2mat(q_handeye2)
    t_handeye2 = [-0.098, -0.075, 0.031]  # m
    t_handeye2 = [i * 1000 for i in t_handeye2]  # mm
    cTb2 = np.identity(4)
    cTb2[:-1, :-1] = R_handeye2
    cTb2[:-1, -1] = t_handeye2
    #print("cTb2\n", cTb2)

    # Florian's on screen stuff, as it is radians, radians, metres, radians ...
    joint_angle_reading_1 = [0.5721309422540486, -0.23263188237026491, 0.15480943325000002, 3.3556681655828435, 0.2705402695355721, 0.0662482860547402, 0.9003524925321471]
    joint_angle_reading_2 = [-0.5470552286114416, -0.23222257752793102, 0.12952011164000002, 2.858283619637551, 0.6248387134273592, -0.09001206424180747, 1.0437685825980445]

    # joint_params = [0]*6
    DaVinci_1 = SerialLinkRobot(logger_level=30)
    joint_params_1 = DaVinci_1.get_joint_angles_from_reading(joint_angle_reading_1)
    final_pose_1, pose_after_joint_4_1, pose_after_joint_5_1, pose_after_joint_6_1,  eHeL_1, eHeR_1 = DaVinci_1.fkin(joint_params_1)

    DaVinci_2 = SerialLinkRobot(logger_level=30)
    joint_params_2 = DaVinci_2.get_joint_angles_from_reading(joint_angle_reading_2)
    final_pose_2, pose_after_joint_4_2, pose_after_joint_5_2, pose_after_joint_6_2,  eHeL_2, eHeR_2 = DaVinci_2.fkin(joint_params_2)

    Tool_tip_transform_joint_4 = np.array([[0.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, -1.0, 0.0],
                                           [1.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0]])

    Tool_tip_transform_joint_5 = np.array([[0.0, -1.0, 0.0, 0.0],
                                           [0.0, 0.0, -1.0, 0.0],
                                           [-1.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0]])

    Tool_tip_transform_joint_6 = np.array([[0.0, -1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0],
                                           [1.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0]])

    Right_hand_to_left_hand_camera = np.array([[1.0, 0.0, 0.0, 0.0],
                                               [0.0, -1.0, 0.0, 0.0],
                                               [0.0, 0.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.0, 1.0]])

    unity_pose_joint_4_1 = Right_hand_to_left_hand_camera @ cTb1 @ pose_after_joint_4_1 @ Tool_tip_transform_joint_4
    quaternions1 = transforms3d.quaternions.mat2quat(unity_pose_joint_4_1[:-1, :-1])  # w, x, y, z
    position1 = unity_pose_joint_4_1[:-1, -1]  # mm
    print("JOINT 4 PSM_1 SHAFT RIGHT ARM Orientation in quaternions  WXYZ", quaternions1)
    print("JOINT 4 PSM_1 SHAFT RIGHT ARM Position in mm", position1)

    unity_pose_joint_4_2 = Right_hand_to_left_hand_camera @ cTb2 @ pose_after_joint_4_2 @ Tool_tip_transform_joint_4
    quaternions2 = transforms3d.quaternions.mat2quat(unity_pose_joint_4_2[:-1, :-1])  # w, x, y, z
    position2 = unity_pose_joint_4_2[:-1, -1]  # mm
    print("JOINT 4 PSM_2 SHAFT LEFT ARM Orientation in quaternions  WXYZ", quaternions2)
    print("JOINT 4 PSM_2 SHAFT LEFT ARM Position in mm", position2)

    unity_pose_joint_5_1 = Right_hand_to_left_hand_camera @ cTb1 @ pose_after_joint_5_1 @ Tool_tip_transform_joint_5
    quaternions3 = transforms3d.quaternions.mat2quat(unity_pose_joint_5_1[:-1, :-1])  # w, x, y, z
    position3 = unity_pose_joint_5_1[:-1, -1]  # mm
    print("JOINT 5 PSM_1 LOGO BODY RIGHT ARM Orientation in quaternions  WXYZ", quaternions3)
    print("JOINT 5 PSM_1 LOGO BODY RIGHT ARM Position in mm", position3)

    unity_pose_joint_5_2 = Right_hand_to_left_hand_camera @ cTb2 @ pose_after_joint_5_2 @ Tool_tip_transform_joint_5
    quaternions4 = transforms3d.quaternions.mat2quat(unity_pose_joint_5_2[:-1, :-1])  # w, x, y, z
    position4 = unity_pose_joint_5_2[:-1, -1]  # mm
    print("JOINT 5 PSM_2 LOGO BODY LEFT ARM Orientation in quaternions  WXYZ", quaternions4)
    print("JOINT 5 PSM_2 LOGO BODY RIGHT ARM Position in mm", position4)

    unity_pose_joint_6_left_jaw_right_arm = Right_hand_to_left_hand_camera @ cTb1 @  pose_after_joint_6_1 @ Tool_tip_transform_joint_6 @eHeL_1
    quaternions5 = transforms3d.quaternions.mat2quat(unity_pose_joint_6_left_jaw_right_arm[:-1, :-1])  # w, x, y, z
    position5 = unity_pose_joint_6_left_jaw_right_arm[:-1, -1]  # mm
    print("JOINT 6 PSM_1 LEFT-JAW RIGHT-ARM Orientation in quaternions  WXYZ", quaternions5)
    print("JOINT 6 PSM_1 LEFT-JAW RIGHT-ARM Position in mm", position5)

    unity_pose_joint_6_right_jaw_right_arm = Right_hand_to_left_hand_camera @ cTb1 @  pose_after_joint_6_1@ Tool_tip_transform_joint_6 @eHeR_1
    quaternions5 = transforms3d.quaternions.mat2quat(unity_pose_joint_6_right_jaw_right_arm[:-1, :-1])  # w, x, y, z
    position5 = unity_pose_joint_6_right_jaw_right_arm[:-1, -1]  # mm
    print("JOINT 6 PSM_1 RIGHT-JAW RIGHT-ARM Orientation in quaternions  WXYZ", quaternions5)
    print("JOINT 6 PSM_1 RIGHT-JAW RIGHT-ARM Position in mm", position5)

    unity_pose_joint_6_left_jaw_left_arm = Right_hand_to_left_hand_camera @ cTb2 @ pose_after_joint_6_2 @ Tool_tip_transform_joint_6@eHeL_2
    quaternions7 = transforms3d.quaternions.mat2quat(unity_pose_joint_6_left_jaw_left_arm[:-1, :-1])  # w, x, y, z
    position7 = unity_pose_joint_6_left_jaw_left_arm[:-1, -1]  # mm
    print("JOINT 6 PSM_2 LEFT-JAW LEFT-ARM Orientation in quaternions  WXYZ", quaternions7)
    print("JOINT 6 PSM_2 LEFT-JAW LEFT-ARM Position in mm", position7)

    unity_pose_joint_6_right_jaw_left_arm = Right_hand_to_left_hand_camera @ cTb2 @ pose_after_joint_6_2 @ Tool_tip_transform_joint_6 @eHeR_2
    quaternions8 = transforms3d.quaternions.mat2quat(unity_pose_joint_6_right_jaw_left_arm[:-1, :-1])  # w, x, y, z
    position8 = unity_pose_joint_6_right_jaw_left_arm[:-1, -1]  # mm
    print("JOINT 6 PSM_2 RIGHT-JAW LEFT-ARM Orientation in quaternions  WXYZ", quaternions8)
    print("JOINT 6 PSM_2 RIGHT-JAW LEFT-ARM Position in mm", position8)


