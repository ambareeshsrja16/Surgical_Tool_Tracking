#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2", Image)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("image_topic", Image, self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # SAVE  cv_image as "img_034.png"


    # MAKE DLC code create predicted_image -> "img_new.png" with points
    predicted_image = "img_new.png"

    # PUBLISH 
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(predicted_image, "bgr8"))
    except CvBridgeError as e:
      print(e)


def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  #cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)