# YOLOv3_Rotation_Grasp_Detection_ROS
<img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/rotation_detection.png" width="75%" height="75%"><img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/chair_pin.gif" width="80%" height="80%">

## Prerequisites
To download the prerequisites for this package (except for ROS itself), navigate to the package folder and run:
```
$ sudo pip install -r requirements.txt
```

## Installation
Navigate to your catkin workspace and run:
```
$ catkin build yolov3_grasp_detection_ros

```

## Basic Usage
1. First, make sure to put your weights in the [models](models) folder. For the **training process** in order to use custom objects, please refer to the original [YOLO page](https://pjreddie.com/darknet/yolo/). As an example, to download pre-trained weights from the COCO data set, go into the [models](models) folder and run:
```
wget http://pjreddie.com/media/files/yolov3.weights
```

2. Modify the parameters in the [launch file](launch/detector.launch) and launch it. You will need to change the `image_topic` parameter to match your camera, and the `weights_name`, `config_name` and `classes_name` parameters depending on what you are trying to do.

## Start yolov3_grasp_detection_ros node
```
$ roslaunch yolov3_grasp_detection_ros detector.launch
```

### Node parameters

* **`image_topic`** (string)

    Subscribed camera topic.

* **`weights_name`** (string)

    Weights to be used from the [models](models) folder.

* **`config_name`** (string)

    The name of the configuration file in the [config](config) folder. Use `yolov3.cfg` for YOLOv3, `yolov3-tiny.cfg` for tiny YOLOv3, and `yolov3-voc.cfg` for YOLOv3-VOC.

* **`classes_name`** (string)

    The name of the file for the detected classes in the [classes](classes) folder. Use `coco.names` for COCO, and `voc.names` for VOC.

* **`publish_image`** (bool)

    Set to true to get the camera image along with the detected bounding boxes, or false otherwise.

* **`detected_objects_topic`** (string)

    Published topic with the detected bounding boxes.

* **`detections_image_topic`** (string)

    Published topic with the detected bounding boxes on top of the image.

* **`confidence`** (float)

    Confidence threshold for detected objects.

### Subscribed topics

* **`image_topic`** (sensor_msgs::Image)

    Subscribed camera topic.

### Published topics    

* **`detected_objects_topic`** (yolov3_pytorch_ros::BoundingBoxes)

    Published topic with the detected bounding boxes.

* **`detections_image_topic`** (sensor_msgs::Image)

    Published topic with the detected bounding boxes on top of the image (only published if `publish_image` is set to true).
