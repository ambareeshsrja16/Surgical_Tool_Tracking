#!/usr/bin/env python
from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import os
from skimage import io
import skimage.color

from deeplabcut.pose_estimation_tensorflow.nnet import predict as ptf_predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
from deeplabcut.utils import auxiliaryfunctions, visualization
import tensorflow as tf

from pathlib import Path
import numpy as np
import time

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


def generate_prediction(MAX_PREDICTION_STEPS = 1000):
    """
    Generator for predicting image
    MAX_PREDICTION_STEPS : Number of predictions that should be done before re-initializing 

    """

    ##################################################
    # Clone arguments from deeplabcut.evaluate_network
    ##################################################

    config = "/root/DLCROS_ws/Surgical_Tool_Tracking/ForwardPassDeepLabCut/DaVinci-Ambar-2019-10-31/config.yaml"
    Shuffles = [1]
    plotting = None
    show_errors = True
    comparisonbodyparts = "all"
    gputouse = None

    # Suppress scientific notation while printing
    np.set_printoptions(suppress=True)


    ##################################################
    # SETUP everything until image prediction
    ##################################################

    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE']  # was potentially set during training

    vers = tf.__version__.split('.')
    if int(vers[0]) == 1 and int(vers[1]) > 12:
        TF = tf.compat.v1
    else:
        TF = tf

    TF.reset_default_graph()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #tf.logging.set_verbosity(tf.logging.WARN)

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
        trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]  # read how many training siterations that corresponds to.

        # name for deeplabcut net (based on its parameters)
        DLCscorer = auxiliaryfunctions.GetScorerName(cfg, shuffle, trainFraction, trainingsiterations)
        print("Running ", DLCscorer, " with # of trainingiterations:", trainingsiterations)

        # Specifying state of model (snapshot / training state)
        sess, inputs, outputs = ptf_predict.setup_pose_prediction(dlc_cfg)

        # Using GPU for prediction
        # Specifying state of model (snapshot / training state)
        # sess, inputs, outputs = ptf_predict.setup_GPUpose_prediction(dlc_cfg)

        print("Analyzing test image ...")
        imagename = "img034.png"
        image = io.imread(imagename, plugin='matplotlib')

        count = 0
        start_time = time.time()
        while count < MAX_PREDICTION_STEPS:

            ##################################################
            # Predict for test image once, and wait for future images to arrive
            ##################################################
            
            print("Calling predict_single_image")
            pose = predict_single_image(image, sess, inputs, outputs, dlc_cfg)

            ##################################################
            # Yield prediction to caller
            ##################################################
            
            image = (yield pose) # Receive image here ( Refer https://stackabuse.com/python-generators/ for sending/receiving in generators)
            
            step_time = time.time()
            print(f"time: {step_time-start_time}")
            start_time = step_time
            count += 1

            if count == MAX_PREDICTION_STEPS:
                print(f"Restart prediction system, Steps have exceeded {MAX_PREDICTION_STEPS}")

        sess.close()  # closes the current tf session
        TF.reset_default_graph()



class image_converter:

  def __init__(self, generator):
    self.image_pub = rospy.Publisher("/dlc_prediction_topic", Image)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/stereo/slave/left/image", Image, self.callback)
    self.generator = generator

  def callback(self,data):
    try:
      self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Generator approach
    try:
        self.points_predicted = self.generator.send(self.cv_image)[:,:2]
    except ValueError as e:
        if str(e) == 'generator already executing':
           print("Prediction ongoing, returning previous image")
           return 

    self.overwrite_image() 

    # PUBLISH 
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)
  
  def overwrite_image(self):
    """For each point in points_predicted make four corners and overwrite those 4 points in the cv_image with blue markers"""

    # TODO: Separate this to another function
    height, width = self.cv_image.shape[:2]

    #Clipping points so that they don't fall outside the image size
    self.points_predicted[:,0] = self.points_predicted[:,0].clip(0, height-1)
    self.points_predicted[:,1] = self.points_predicted[:,1].clip(0, width-1)
    
    # Prepare 4 corners for each point => (floor, floor), (ceil, ceil), (floor, ceil), (ceil, floor)
    corner_1 = np.floor(np.copy(self.points_predicted)).astype(int) # (10,2)  
    corner_2 = np.ceil(np.copy(self.points_predicted)).astype(int)  # (10,2)

    corner_3 = np.copy(self.points_predicted) # (10,2)
    corner_3[:,0] = np.floor(corner_3[:,0])
    corner_3[:,1] = np.ceil(corner_3[:,1])
    corner_3 = corner_3.astype(int)

    corner_4 = np.copy(self.points_predicted) # (10,2)
    corner_4[:,0] = np.ceil(corner_4[:,0])
    corner_4[:,1] = np.floor(corner_4[:,1])
    corner_4 = corner_4.astype(int)
    
    # Change those 4 corners to blue (0, 0, 255) (R,G,B)
    for corner in (corner_1, corner_2, corner_3, corner_4):
        for point in range(len(self.points_predicted)):
            self.cv_image[tuple(corner[point].tolist()+[0])] = 0
            self.cv_image[tuple(corner[point].tolist()+[1])] = 0
            self.cv_image[tuple(corner[point].tolist()+[2])] = 255


def main(args):
  MAX_PREDICTION_STEPS = int(1e5)

  # Initialize and kickstart generator to test the first saved image
  generator = generate_prediction(MAX_PREDICTION_STEPS)
  points_predicted = generator.send(None)
  print(f"First prediction: {points_predicted}")

  ic = image_converter(generator)
  rospy.init_node('image_converter', anonymous=True)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
