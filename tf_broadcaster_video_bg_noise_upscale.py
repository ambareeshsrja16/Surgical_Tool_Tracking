#!/usr/bin/env python
import rospy

import tf
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header

import numpy as np
import cv2
from  cv_bridge import CvBridge, CvBridgeError

def main(from_frame_name ,
        to_frame_name ,
        video_filename ,
        freq_in_hz ,
        variance_scale,
        frame_size,
        noise_variance,):
    """
    to_T_from => Transformation
        
    from_frame_name = '/randomized_PSM1_base'
    to_frame_name = '/cam0'
    freq_in_hz =1
    """

    rospy.init_node("tf_broadcaster") 
    br = tf.TransformBroadcaster()
    
    translation = (0.100, -0.061, 0.008)  #Copying values from TF monitor
    rotation_quart = (-0.294, 0.844, -0.434, -0.117) # Quaternion values, read from TF monitor

    pub_joint = rospy.Publisher('joint_state_randomized', JointState, queue_size=10) #Name of topic published - matching with config.json
    pub_image = rospy.Publisher('bg_image_randomized', Image, queue_size=10) #Name of topic published - matching with config.json

    init_position = [0.743583334794707, -0.28510785225062046, 0.14844724193, 2.5263444391015493, 0.02760057295261897, 0.6878535864557926, 0.9991756525433861]
    cam = cv2.VideoCapture(video_filename)

    rate = rospy.Rate(freq_in_hz)

    while not rospy.is_shutdown() and cam.isOpened(): #TODO Check if isOpened() is necessary?

        ret_val, frame = cam.read() #TODO Should a check be added for ret_value=False?

        # Resize Image
        frame = cv2.resize(frame, frame_size)

        # Add Noise
        frame = gaussian_noise(frame, noise_variance)

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ""

        msg_image_frame = CvBridge().cv2_to_imgmsg(frame,"bgr8")
        msg_image_frame.header.stamp = msg.header.stamp
        msg_image_frame.header.frame_id = "" #TODO Why is this left empty? Also, is the default empty, making this line unnecessary?

        msg.position =  [np.random.normal(mean, abs(variance_scale*mean)) for mean in init_position[:3]]+[np.random.normal(mean, abs(10*variance_scale*mean))for mean in init_position[3:]] #Last three joints can have higher variance
        msg.velocity = [] 
        msg.effort = []
                
        pub_joint.publish(msg)
        pub_image.publish(msg_image_frame)
        br.sendTransform(translation,
                rotation_quart,
                msg.header.stamp, #in place of rospy.Time.now(), so that both br.sendTransform and JointState message have the same msg.header.stamp
                from_frame_name,
                to_frame_name)
        
        rospy.loginfo("Publishing background image, transform, and randomized joint state at {rate} Hz, Variance Scale at {var_scale}".format(rate=freq_in_hz, var_scale=variance_scale))

        # DEBUG
        print "msg_image_frame type"
        print  str(msg_image_frame.height)
        print  str(msg_image_frame.width)
        print  str(msg_image_frame.encoding)
        # DEBUG

        rate.sleep()            



def gaussian_noise(image, sigma):
    """
    https://stackoverflow.com/a/57489530/9652400

    image: read through PIL.Image.open('path')
    sigma: variance of Standard Gaussian noise

    IMPORTANT: when reading a image into numpy arrary, the default dtype is uint8,
    which can cause wrapping when adding noise onto the image. 
    E.g,  example = np.array([128,240,255], dtype='uint8')
         example + 50 = np.array([178,44,49], dtype='uint8')
    Transfer np.array to dtype='int16' can solve this problem.
    """
    import numpy as np 
    from PIL import Image as PIL_Image

    image = PIL_Image.fromarray(image)

    img = np.array(image)
    noise = np.random.randn(img.shape[0], img.shape[1], img.shape[2])
    img = img.astype('int16')
    img_noise = img + noise * sigma
    img_noise = np.clip(img_noise, 0, 255)
    img_noise = img_noise.astype('uint8')
    final_image = PIL_Image.fromarray(img_noise)
    
    #final_image = cv2.cvtColor(np.asarray(final_image), cv2.COLOR_RGB2BGR)
    final_image = np.asarray(final_image) 

    return final_image


if __name__ == '__main__':

    from_frame_name = '/randomized_PSM1_base'
    to_frame_name = '/cam0'
    video_filename = "/home/ambareesh/Research/endo_video.mov" #Same location as rosbag
    freq_in_hz = 1 
    variance_scale = 0.1
    REQ_SIZE = (1920, 1080)
    
    #NOISE_VARIANCE = 100
    NOISE_VARIANCE = 10
    #NOISE_VARIANCE = 1
    #NOISE_VARIANCE = 0

    main(from_frame_name,
        to_frame_name,
        video_filename,
        freq_in_hz,
        variance_scale,
        frame_size = REQ_SIZE,
        noise_variance = NOISE_VARIANCE,)

