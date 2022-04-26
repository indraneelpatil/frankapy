# Created by Indraneel and Shivani on 25/04/22

import cv2
from cv_bridge import CvBridge

import numpy as np
import rospy
from utils import *
from perception import CameraIntrinsics
from utils import *
from RobotUtil import *


if __name__ == '__main__':
    rospy.init_node('toss_bot_perception', anonymous=True)
    print('Opening the eyes!!')

    rospy.sleep(1)
    AZURE_KINECT_INTRINSICS = 'calib/azure_kinect.intr'
    AZURE_KINECT_EXTRINSICS = 'calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf'
    
    cv_bridge = CvBridge()
    azure_kinect_intrinsics = CameraIntrinsics.load(AZURE_KINECT_INTRINSICS)
    azure_kinect_to_world_transform = RigidTransform.load(AZURE_KINECT_EXTRINSICS)    
    print('Reached here!')
    azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    print('Reached here 2!')
    azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
    
    frame = azure_kinect_rgb_image[:,:,:3]

    # Normalise image
    img_mag = np.linalg.norm(img, axis=2)
    img_norm = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_norm[i,j] = img[i,j] / img_mag[i,j]
    
    cv2.namedWindow("image")
    cv2.imshow("image", frame)
    #cv2.setMouseCallback("image", onMouse, object_image_position)
    cv2.waitKey()
    cv2.destroyAllWindows()
