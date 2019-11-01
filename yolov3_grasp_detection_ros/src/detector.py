#!/usr/bin/env python

from __future__ import division

# Python imports
import numpy as np
import scipy.io as sio
import os, sys, cv2, time
from skimage.transform import resize

# ROS imports
import rospy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
from yolov3_pytorch_ros.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge, CvBridgeError

package = RosPack()
package_path = package.get_path('yolov3_grasp_detection_ros')

# Deep learning imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")
# Detector manager class for YOLO
class DetectorManager():
    def __init__(self):
        # Load weights parameter
        weights_name = rospy.get_param('~weights_name', 'yolov3.weights')
        self.weights_path = os.path.join(package_path, 'models', weights_name)
        rospy.loginfo("Found weights, loading %s", self.weights_path)

        # Raise error if it cannot find the model
        if not os.path.isfile(self.weights_path):
            raise IOError(('{:s} not found.').format(self.weights_path))

        # Load image parameter and confidence threshold
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
        self.confidence_th = rospy.get_param('~confidence', 0.7)
        self.nms_th = rospy.get_param('~nms_th', 0.3)

        # Load publisher topics
        self.detected_objects_topic = rospy.get_param('~detected_objects_topic')
        self.published_image_topic = rospy.get_param('~detections_image_topic')

        # Load other parameters
        config_name = rospy.get_param('~config_name', 'yolov3.cfg')
        self.config_path = os.path.join(package_path, 'config', config_name)
        classes_name = rospy.get_param('~classes_name', 'coco.names')
        self.classes_path = os.path.join(package_path, 'classes', classes_name)
        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.network_img_size = rospy.get_param('~img_size', 608)
        self.publish_image = rospy.get_param('~publish_image')

        # Initialize width and height
        self.h = 0
        self.w = 0

        # Load net
        self.model = Darknet(self.config_path, img_size=self.network_img_size)
        self.model.load_weights(self.weights_path)
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            raise IOError('CUDA not found.')
        self.model.eval() # Set in evaluation mode
        rospy.loginfo("Deep neural network loaded")

        # Load CvBridge
        self.bridge = CvBridge()

        # Load classes
        self.classes = load_classes(self.classes_path) # Extracts class labels from file
        self.classes_colors = {}

        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCb, queue_size = 1, buff_size = 2**24)

        # Define publishers
        self.pub_ = rospy.Publisher(self.detected_objects_topic, BoundingBoxes, queue_size=10)
        self.pub_viz_ = rospy.Publisher(self.published_image_topic, Image, queue_size=10)
        rospy.loginfo("Launched node for object detection")

        # Spin
        rospy.spin()

    def imageCb(self, data):
        # Convert the image to OpenCV
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        # Initialize detection results
        detection_results = BoundingBoxes()
        detection_results.header = data.header
        detection_results.image_header = data.header

        # Configure input
        input_img = self.imagePreProcessing(self.cv_image)
        input_img = Variable(input_img.type(torch.cuda.FloatTensor))

        # Get detections from network
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, 80, self.confidence_th, self.nms_th)

        # Parse detections
        if detections[0] is not None:
            for detection in detections[0]:
                # Get xmin, ymin, xmax, ymax, confidence and class
                xmin, ymin, xmax, ymax, _, conf, det_class = detection
                pad_x = max(self.h - self.w, 0) * (self.network_img_size/max(self.h, self.w))
                pad_y = max(self.w - self.h, 0) * (self.network_img_size/max(self.h, self.w))
                unpad_h = self.network_img_size-pad_y
                unpad_w = self.network_img_size-pad_x
                xmin_unpad = ((xmin-pad_x//2)/unpad_w)*self.w
                xmax_unpad = ((xmax-xmin)/unpad_w)*self.w + xmin_unpad
                ymin_unpad = ((ymin-pad_y//2)/unpad_h)*self.h
                ymax_unpad = ((ymax-ymin)/unpad_h)*self.h + ymin_unpad
                w = xmax_unpad - xmin_unpad
                h = ymax_unpad - ymin_unpad

                # Populate darknet message
                detection_msg = BoundingBox()
                detection_msg.xmin = xmin_unpad
                detection_msg.xmax = xmax_unpad
                detection_msg.ymin = ymin_unpad
                detection_msg.ymax = ymax_unpad
                #detection_msg.cx = (xmax_unpad - xmin_unpad)/2 + xmin_unpad
                #detection_msg.cy = (ymax_unpad - ymin_unpad)/2 + ymin_unpad
                if w > h:
                    detection_msg.degree = True
                    detection_msg.cx = (((xmax_unpad - xmin_unpad)/2 + xmin_unpad))
                    detection_msg.cy = (ymax_unpad - ymin_unpad)/2 + ymin_unpad
                    #print(detection_msg.cx,detection_msg.cy)

                else:
                    detection_msg.degree = False
                    detection_msg.cx = (xmax_unpad - xmin_unpad)/2 + xmin_unpad
                    detection_msg.cy = (((ymax_unpad - ymin_unpad)/2 + ymin_unpad))
                    #print(detection_msg.cx,detection_msg.cy)

                detection_msg.probability = conf
                detection_msg.Class = self.classes[int(det_class)]

                # Append in overall detection message
                detection_results.bounding_boxes.append(detection_msg)

        # Publish detection results
        self.pub_.publish(detection_results)

        # Visualize detection results
        if (self.publish_image):
            self.visualizeAndPublish(detection_results, self.cv_image)
        return True


    def imagePreProcessing(self, img):
        # Extract image and shape
        img = np.copy(img)
        img = img.astype(float)
        height, width, channels = img.shape

        if (height != self.h) or (width != self.w):
            self.h = height
            self.w = width

            # Determine image to be used
            self.padded_image = np.zeros((max(self.h,self.w), max(self.h,self.w), channels)).astype(float)

        # Add padding
        if (self.w > self.h):
            self.padded_image[(self.w-self.h)//2 : self.h + (self.w-self.h)//2, :, :] = img
        else:
            self.padded_image[:, (self.h-self.w)//2 : self.w + (self.h-self.w)//2, :] = img

        # Resize and normalize
        input_img = resize(self.padded_image, (self.network_img_size, self.network_img_size, 3))/255.

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        input_img = input_img[None]

        return input_img


    def visualizeAndPublish(self, output, imgIn):
        # Copy image and visualize
        imgOut = imgIn.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        thickness = 1
        point_color = (0,255,255)
        #imgOut_1 = imgIn.copy()
        imgray = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 90, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for index in range(len(output.bounding_boxes)):
            label = output.bounding_boxes[index].Class
            x_p1 = output.bounding_boxes[index].xmin
            y_p1 = output.bounding_boxes[index].ymin
            x_p3 = output.bounding_boxes[index].xmax
            y_p3 = output.bounding_boxes[index].ymax
            confidence = output.bounding_boxes[index].probability
            w = int(x_p3 - x_p1)
            h = int(y_p3 - y_p1)
            center = (int(((x_p1)+(x_p3))/2)+5,int(((y_p1)+(y_p3))/2))
            #print(center)
            '''
            if w > h:
                center = (int(((x_p1)+(x_p3))/2)+5,int(((y_p1)+(y_p3))/2))
                cv2.rectangle(imgOut, (int(center[0])-5, int(center[1])-20), (int(center[0])+5, int(center[1])+20), (0,255,0),1)
                cv2.circle(imgOut,(center), 2, (255,255,0), 4)

            else:
                center = (int(((x_p1)+(x_p3))/2),int(((y_p1)+(y_p3))/2)-5)
                cv2.rectangle(imgOut, (int(center[0])+20, int(center[1])+5), (int(center[0])-20, int(center[1])-5), (0,255,0),1)
                cv2.circle(imgOut,(center), 2, (255,255,0), 4)
            '''

            #cv2.circle(img, (580,350), 2, point_color, 4)

            # Find class color
            if label in self.classes_colors.keys():
                color = self.classes_colors[label]
            else:
                # Generate a new color if first time seen this label
                color = np.random.randint(0,198,3)
                self.classes_colors[label] = color

            # Create rectangle
            #cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(y_p3)), (color),2)
            #print(color)
            text = ('{:s}: {:.2f}').format(label,confidence)
            cv2.putText(imgOut, text, (int(((x_p1)+(x_p3))/2)+5,int(((y_p1)+(y_p3))/2)), font, fontScale, (0,0,0), thickness ,cv2.LINE_AA)

        # Rotation
        for i in range(len(contours)):
            if (contours[i][0][0][0]) >= 480 or (contours[i][0][0][1]) >= 580 or (contours[i][0][0][0]) <= 10:
                continue

            cnt = contours[i]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #detection_msg = box()
            cx = int(rect[0][0])
            cy = int(rect[0][1])

            if rect[1][0] > rect[1][1]:
                cv2.putText(imgOut, "left" +" "+ str("%.2f"%(rect[2])), (cx+10,cy-10),font, 0.3, (255,255,255), 1)

            if rect[1][0] <= rect[1][1]:
                cv2.putText(imgOut, "right" +" "+ str("%.2f"%(90 + rect[2])), (cx+10,cy-10),font, 0.3, (255,255,255), 1)

            cv2.drawContours(imgOut,[box],0,(230,0,0),1)

            #print(box)
            #cv2.line(imgOut,(cx+20,cy),(cx-20,cy),(0,0,0),1)
            #cv2.drawContours(imgOut,[box],0,(0,0,255),2)
            cv2.circle(imgOut, (cx,cy), 1, point_color, 1)
            # Create center
            #cv2.circle(imgOut,(center), 2, (255,255,0), 4)


        # Publish visualization image
        image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
        self.pub_viz_.publish(image_msg)


if __name__=="__main__":
    # Initialize node
    rospy.init_node("detector_manager_node")

    # Define detector object
    dm = DetectorManager()
