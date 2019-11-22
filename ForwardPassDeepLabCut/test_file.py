import deeplabcut

if __name__ == '__main__':
    # task = 'DaVinci'  # Enter the name of your experiment Task
    # experimenter = 'Ambar'  # Enter the name of the experimenter
    # video = ["C:/Users/asree/PycharmProjects/ForwardPassDeepLabCut/videos_training/video_1.avi"]  # Enter the paths of your videos OR FOLDER you want to grab frames from.
    # path_config_file = deeplabcut.create_new_project(task, experimenter, video, copy_videos=True)

    # NOTE: The function returns the path, where your project is.
    # You could also enter this manually (e.g. if the project is already created and you want to pick up, where you stopped...)
    # path_config_file = '/home/Mackenzie/Reaching/config.yaml' # Enter the path of the config file that was just created from the above step (check the folder)
    # deeplabcut.create_training_dataset(path_config_file)

    path_config_file = "/root/DLCROS_ws/Surgical_Tool_Tracking/ForwardPassDeepLabCut/DaVinci-Ambar-2019-10-31/config.yaml"
    deeplabcut.evaluate_network(path_config_file, plotting=True)
