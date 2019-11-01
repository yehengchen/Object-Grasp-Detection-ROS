# screw_grasp_detection

<img src="https://github.com/yehengchen/YOLOv3_ROS/blob/master/yolov3_pytorch_ros/models/screw_gazebo.png" width="80%" height="70%">

The package has been tested with Ubuntu 16.04/18.04 and ROS Kinetic/Melodic on a NVIDIA GTX 1080Ti.

## Prerequisites
To download the prerequisites for this package (except for ROS itself), navigate to the package folder and run:
```
$ sudo pip install -r requirements.txt
```

## Installation
Navigate to your catkin workspace and run:
```
$ catkin build yolov3_pytorch_ros
```
## Basic Usage
1. First, make sure to put your weights in the [models](models) folder. For the **training process** in order to use custom objects, please refer to the original [YOLO page](https://pjreddie.com/darknet/yolo/). As an example, to download pre-trained weights from the COCO data set, go into the [models](models) folder and run:
```
wget http://pjreddie.com/media/files/yolov3.weights
```

2. Modify the parameters in the [launch file](launch/detector.launch) and launch it. You will need to change the `image_topic` parameter to match your camera, and the `weights_name`, `config_name` and `classes_name` parameters depending on what you are trying to do.

## Start yolov3_pytorch_ros node
```
$ roslaunch yolov3_pytorch_ros detector.launch
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

## Citing

The YOLO methods used in this software are described in the paper: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640).

If you are using this package, please add the following citation to your publication:

    @misc{vasilopoulos_pavlakos_yolov3ros_2019,
      author = {Vasileios Vasilopoulos and Georgios Pavlakos},
      title = {{yolov3_pytorch_ros}: Object Detection for {ROS} using {PyTorch}},
      howpublished = {\url{https://github.com/vvasilo/yolov3_pytorch_ros}},
      year = {2019},
    }
