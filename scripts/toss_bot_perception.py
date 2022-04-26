# Created by Indraneel and Shivani on 25/04/22

import cv2
from cv_bridge import CvBridge

import numpy as np
import rospy
from perception import CameraIntrinsics
from utils import *
#from RobotUtil import *


if __name__ == '__main__':
    rospy.init_node('toss_bot_perception', anonymous=True)
    print('Opening the eyes!!')

    rospy.sleep(1)
    AZURE_KINECT_INTRINSICS = 'calib/azure_kinect.intr'
    AZURE_KINECT_EXTRINSICS = 'calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf'
    
    cv_bridge = CvBridge()
    azure_kinect_intrinsics = CameraIntrinsics.load(AZURE_KINECT_INTRINSICS)
    azure_kinect_to_world_transform = RigidTransform.load(AZURE_KINECT_EXTRINSICS)    
    
    while not rospy.is_shutdown():
        azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
        azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
        
        frame = azure_kinect_rgb_image[:,:,:3]

        # Normalise image
        img_mag = np.linalg.norm(frame, axis=2)
        img_norm = np.zeros(frame.shape)
        img_norm = frame / np.expand_dims(img_mag,2)

        # Thresholding and masking
        red_thresh = 0.8
        red_mask = ((img_norm[:,:,2] > red_thresh)*255).astype('uint8')

        # set kernel for morphology stuff
        kernel = np.ones((3,3),np.uint8)

        # closes small patches
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # open image 
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # find contours 
        contours_red, _ = cv2.findContours(image=red_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # find the biggest countour (c) by the area
        c = max(contours_red, key = cv2.contourArea)

         # Convert contour into minimum area rectangle
        rect = cv2.minAreaRect(c)

        # Convert minimum area rectangle into four points
        pts = cv2.boxPoints(rect)

        # Use longer edge to determine angle
        center = np.mean(pts,axis=0)
        #print(center)

        # draw center on image
        azure_kinect_rgb_image = cv2.circle(azure_kinect_rgb_image, (int(center[0]),int(center[1])), radius=10, color=(255, 255, 255), thickness=-1)

        # draw the biggest contour (c) in green
        #cv2.rectangle(red_mask,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(azure_kinect_rgb_image, c, -1, (0,0,0), 20)

        
        cv2.namedWindow("image")
        azure_kinect_rgb_image = cv2.resize(azure_kinect_rgb_image, (1160, 740))  
        cv2.imshow("image", azure_kinect_rgb_image)
        #cv2.setMouseCallback("image", onMouse, object_image_position)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
