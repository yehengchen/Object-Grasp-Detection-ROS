# Real-time Object Detection for ROS 

![](https://github.com/yehengchen/yolov3_ros/blob/master/ros_yolov3_tiny.gif)

## Development Environment

- Ubuntu 16.04.2 
- ROS Kinetic
- OpenCV

***

## Quick start

In order to install darknet_ros, clone the latest version using SSH (see how to set up an SSH key) from this repository into your catkin workspace and compile the package using ROS.

    cd catkin_workspace/src
    git clone https://github.com/yehengchen/yolov3_ros.git
    cd ../
    
To maximize performance, make sure to build in Release mode. You can specify the build type by setting

    catkin_make -DCMAKE_BUILD_TYPE=Release

***

## Download weights

The yolo-voc.weights and tiny-yolo-voc.weights are downloaded automatically in the CMakeLists.txt file. If you need to download them again, go into the weights folder and download the two pre-trained weights from the COCO data set:

    cd catkin_workspace/src/darknet_ros/darknet_ros/yolo_network_config/weights/

* __COCO data set (Yolo v2):__

      wget http://pjreddie.com/media/files/yolov2.weights
      wget http://pjreddie.com/media/files/yolov2-tiny.weights

* __VOC data set (Yolo v2):__
        
      wget http://pjreddie.com/media/files/yolov2-voc.weights
      wget http://pjreddie.com/media/files/yolov2-tiny-voc.weights

* __Yolov3:__
        
      wget http://pjreddie.com/media/files/yolov3.weights
      wget http://pjreddie.com/media/files/yolov3-voc.weights

***

## Use your own detection objects

In order to use your own detection objects you need to provide your weights and your cfg file inside the directories:

    catkin_workspace/src/darknet_ros/darknet_ros/yolo_network_config/weights/
    catkin_workspace/src/darknet_ros/darknet_ros/yolo_network_config/cfg/

***

## Running Node

__Launch darknet_ros with usb_cam:__

    roslaunch usb_cam usb_cam-test.launch

    roslaunch darknet_ros darknet_ros.launch

    roslaunch darknet_ros yolo_v3.launch 

__The node will publish the following 3 topics__
    
    rostopic echo /darknet_ros/bounding_boxes
    rostopic echo /darknet_ros/found_object
    rostopic echo /darknet_ros/detection_image

***

## Node

ROS related parameters

You can change the names and other parameters of the publishers, subscribers and actions inside darkned_ros/config/ros.yaml.

__Published Topics__

* __found_object ([std_msgs::Int8])__

Publishes the number of detected objects.

* __bounding_boxes ([darknet_ros_msgs::BoundingBoxes])__

Publishes an array of bounding boxes that gives information of the position and size of the bounding box in pixel coordinates.

* __detection_image ([sensor_msgs::Image])__

Publishes an image of the detection image including the bounding boxes.

***

## Citing

YOLOv3 :

    @article{yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Redmon, Joseph and Farhadi, Ali},
    journal = {arXiv},
    year={2018}
    }
## Reference

@mbjelonic darknet_ros [GitHub](https://github.com/leggedrobotics/darknet_ros)



