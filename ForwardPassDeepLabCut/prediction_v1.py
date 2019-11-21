import numpy as np

def predict_single_image(image, sess, inputs, outputs, dlc_cfg):
    """
    Returns pose for one single image
    :param image:
    :return:
    """
    # assert
    image = skimage.color.gray2rgb(image)
    image_batch = data_to_input(image)

    # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = ptf_predict.extract_cnn_output(outputs_np, dlc_cfg)

    # Extract maximum scoring location from the heatmap, assume 1 person
    pose = ptf_predict.argmax_pose_predict(scmap, locref, dlc_cfg.stride)

    return pose


if __name__ == '__main__':

    ##################################################
    # Clone arguments to deeplabcut.evaluate_network
    ##################################################

    config = "/root/DLCROS_ws/Surgical_Tool_Tracking/ForwardPassDeepLabCut/DaVinci-Ambar-2019-10-31/config.yaml"
    Shuffles = [1]
    plotting = None
    show_errors = True
    comparisonbodyparts = "all"
    gputouse = None

    # Suppress scientific notation while printing
    np.set_printoptions(suppress=True)

    # Check time stamp differences between events
    import time


    ##################################################
    # SETUP everything until image prediction
    ##################################################

    # Clone evaluate_network
    import os
    from skimage import io
    import skimage.color

    from deeplabcut.pose_estimation_tensorflow.nnet import predict as ptf_predict
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
    from deeplabcut.utils import auxiliaryfunctions, visualization
    import tensorflow as tf

    from pathlib import Path

    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE']  # was potentially set during training

    vers = tf.__version__.split('.')
    if int(vers[0]) == 1 and int(vers[1]) > 12:
        TF = tf.compat.v1
    else:
        TF = tf

    TF.reset_default_graph()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #
    #    tf.logging.set_verbosity(tf.logging.WARN)

    start_path = os.getcwd()
    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)
    if gputouse is not None:  # gpu selectinon
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

    ##############
    # Cloning for-loop variables
    shuffle = Shuffles[0]
    trainFraction = cfg["TrainingFraction"][0]
    ##############

    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
    # Get list of body parts to evaluate network for
    comparisonbodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg, comparisonbodyparts)

    ##################################################
    # Load and setup CNN part detector
    ##################################################
    modelfolder = os.path.join(cfg["project_path"], str(auxiliaryfunctions.GetModelFolder(trainFraction, shuffle, cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    # Load meta data
    # data, trainIndices, testIndices, trainFraction = auxiliaryfunctions.LoadMetadata(
    #     os.path.join(cfg["project_path"], metadatafn))

    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle s and trainFraction %s does not exist.")

    dlc_cfg['batch_size'] = 1  # in case this was edited for analysis.

    # Check which snapshots are available and sort them by # iterations
    Snapshots = np.array(
        [fn.split('.')[0] for fn in os.listdir(os.path.join(str(modelfolder), 'train')) if "index" in fn])
    try:  # check if any where found?
        Snapshots[0]
    except IndexError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle and "
            "trainFraction is not trained.\nPlease train it before evaluating."
            "\nUse the function 'train_network' to do so.")

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    if cfg["snapshotindex"] == -1:
        snapindices = [-1]
    elif cfg["snapshotindex"] == "all":
        snapindices = range(len(Snapshots))
    elif cfg["snapshotindex"] < len(Snapshots):
        snapindices = [cfg["snapshotindex"]]
    else:
        print("Invalid choice, only -1 (last), any integer up to last, or all (as string)!")

    ##################################################
    # Compute predictions over image
    ##################################################
    for snapindex in snapindices:
        dlc_cfg['init_weights'] = os.path.join(str(modelfolder), 'train',
                                               Snapshots[snapindex])  # setting weights to corresponding snapshot.
        trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[
            -1]  # read how many training siterations that corresponds to.

        # name for deeplabcut net (based on its parameters)
        DLCscorer = auxiliaryfunctions.GetScorerName(cfg, shuffle, trainFraction, trainingsiterations)
        print("Running ", DLCscorer, " with # of trainingiterations:", trainingsiterations)

        # Specifying state of model (snapshot / training state)
        sess, inputs, outputs = ptf_predict.setup_pose_prediction(dlc_cfg)

        # Using GPU for prediction
        # Specifying state of model (snapshot / training state)
        # sess, inputs, outputs = ptf_predict.setup_GPUpose_prediction(dlc_cfg)

        print("Analyzing image...")

        ##################################################
        # Wait for image to arrive
        ##################################################

        image_arrived = True
        # dikeo
        imagename = "img034.PNG"

        count = 0
        init = time.time()
        start = time.time()
        while image_arrived:
            # DIKEO
            # Equivalent of below line, but from a port/ROS node
            #image = io.imread(imagename, mode='RGB')
            image = io.imread(imagename, plugin='matplotlib')

            ##################################################
            # Once image arrives,call the function that uses the setup to predict and return 10*3 nd.array
            ##################################################
            # DEBUG
            print("Calling predict_single image")
            pose = predict_single_image(image, sess, inputs, outputs, dlc_cfg)

            ##################################################
            # Send prediction to output stream
            ##################################################

            if not count:
                print("Pose: image 1")
                print("\n",pose)

            done = time.time()
            print(f"pose evaluated for image {count+1}, took {done-start} seconds")
            start = done
            count += 1


            # dikeo
            if count > 100:
                print(f"Took {(done-init):.2f} seconds to print {count} images)")
                break
        sess.close()  # closes the current tf session
        TF.reset_default_graph()
