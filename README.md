# Object-Grasp-Detection-ROS

## Development Environment

- __Ubuntu 16.04 / 18.04__
- __ROS Kinetic / Melodic__
- __OpenCV__

## ROS Installation Options
[ROS (Robot Operating System)](http://wiki.ros.org)
* __Ubuntu install of ROS [Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)__
* __Ubuntu install of ROS [Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)__

***
## Real-time Grasp (Rotation Angle) Detection With ROS
<img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/chair_pin.gif" width="75%" height="75%">

__Gazebo Real-time Screw Rotation Detection - [[Link]](https://github.com/yehengchen/YOLOv3-ROS/tree/master/yolov3_grasp_detection_ros)__


***
## Real-time Screw Grasp Detection With ROS
<img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/grasp_detection.gif" width="75%" height="75%">

__Gazebo Real-time Grasp Detection - [[Link]](https://github.com/yehengchen/YOLOv3-ROS/tree/master/yolov3_pytorch_ros)__

__Parts-Arrangement-Robot - [[Link]](https://github.com/Kminseo/Parts-Arrangement-Robot)__

***
## Real-time Screw Detection With ROS
![](https://github.com/yehengchen/YOLOv3-ROS/blob/master/darknet_ros/yolo_network_config/weights/output.gif)


__Gazebo Real-time Screw Grasp Detection - [[Link]](https://github.com/yehengchen/YOLOv3-ROS/tree/master/darknet_ros)__


# YOLOv3_ROS object detection

## Prerequisites
To download the prerequisites for this package (except for ROS itself), navigate to the package folder and run:

```
$ cd yolov3_pytorch_ros
$ sudo pip install -r requirements.txt
```

## Installation
Navigate to your catkin workspace and run:
```
$ catkin_make yolov3_pytorch_ros
```
## Basic Usage
1. First, make sure to put your weights in the [models](models) folder. For the **training process** in order to use custom objects, please refer to the original [YOLO page](https://pjreddie.com/darknet/yolo/). As an example, to download pre-trained weights from the COCO data set, go into the [models](models) folder and run:
```
wget http://pjreddie.com/media/files/yolov3.weights
```

2. Modify the parameters in the [launch file](launch/detector.launch) and launch it. You will need to change the `image_topic` parameter to match your camera, and the `weights_name`, `config_name` and `classes_name` parameters depending on what you are trying to do.

## Start yolov3 pytorch ros node
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
    
---
# Ubuntu-18.04 Realsense D435
* ### The steps are described in bellow documentation
  __[[IntelRealSense -Linux Distribution]](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)__
  
  ```
  
  sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key
  
  sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" -u
  
  sudo apt-get install librealsense2-dkms
  
  sudo apt-get install librealsense2-utils
  
  sudo apt-get install librealsense2-dev
  
  sudo apt-get install librealsense2-dbg #(리얼센스 패키지 설치 확인하기)
  
  realsense-viewer
  
  ```
  
* ### Installing Realsense-ros 
  1) __catkin workspace__
  
  ```
  mkdir -p ~/catkin_ws/src
  cd ~/catkin_ws/src/
  ```
  
  2) __Download realsense-ros pkg__
  	
  ```
  git clone https://github.com/IntelRealSense/realsense-ros.git
  cd realsense-ros/
  git checkout `git tag | sort -V | grep -P "^\d+\.\d+\.\d+" | tail -1`
  cd ..
  ```
  
  3) __Download ddynamic_reconfigure__
  
  ```
  cd src
  git clone https://github.com/pal-robotics/ddynamic_reconfigure/tree/kinetic-devel
  cd ..
  ```
  
  4) __Pkg installation__
  
  ```
  catkin_init_workspace
  cd ..
  catkin_make clean
  catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
  catkin_make install
  echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
  source ~/.bashrc
  ```
  

  5) __Run D435 node__
  ```
  roslaunch realsense2_camera rs_camera.launch
  ```
  
  6) __Run rviz testing__

  ```
  rosrun rviz rvzi
  Add > Image to view the raw RGB image
  ```

---
# How to train (to detect your custom objects)

__Training YOlOv3:__
- __[[How to make custom dataset for yolov3]](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo/yolov3)__

### Download the dakrnet source code
    git clone https://github.com/pjreddie/darknet
    cd darknet
    
    vim Makefile
    ...
	GPU=1 # if no using GPU 0
	CUDNN=1 # if no 0
	OPENCV=0
	OPENMP=0
	DEBUG=0
    
    make
##### 0. Create folder for yolov3    
    mkdir yolov3
    cd yolov3
    mkdir JPEGImages labels backup cfg 


__├── [JPEGImages](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/JPEGImages) <br>__
│   ├── object-00001.jpg <br>
│   └── object-00002.jpg <br>
│   ... <br>
__├── [labels](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/labels) <br>__
│   ├── object-00001.txt <br>
│   └── object-00002.txt <br>
│   ... <br>
__├── [backup](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/backup) <br>__
│   ├── yolov3-object.backup <br>
│   └── yolov3-object_20000.weights <br>
│   ... <br>
__├── [cfg](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/cfg) <br>__
│   ├── obj.data <br>
│   ├── yolo-obj.cfg  <br>
│   └── obj.names <br>
└── obj_test.txt...
    
##### 1. Create file `yolo-obj.cfg` with the same content as in `yolov3.cfg` (or copy `yolov3.cfg` to `yolo-obj.cfg)` and:

  * change line batch to [`batch=64`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L3)
  * change line subdivisions to [`subdivisions=8`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)
  * change line max_batches to (`classes*2000` but not less than `4000`), f.e. [`max_batches=6000`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L20) if you train for 3 classes
  * change line steps to 80% and 90% of max_batches, f.e. [`steps=4800,5400`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L22)
  * change line `classes=80` to your number of objects in each of 3 
 	`[yolo]`-layers:
		
      * cfg/yolov3.cfg#L610
      * cfg/yolov3.cfg#L696
      * cfg/yolov3.cfg#L783
	
		```
		[convolutional]
		...
		filters = 24 #3*(classes + 5)
		[yolo]
		...
		classes=3
		```
	
  * change [`filters=255`] to filters= `3x(classes + 5)` in the 3 `[convolutional]` before each `[yolo]` layer
      * cfg/yolov3.cfg#L603
      * cfg/yolov3.cfg#L689
      * cfg/yolov3.cfg#L776

  So if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`.
  
  **(Do not write in the cfg-file: filters=(classes + 5)x3)**
  

##### 2. Create file `obj.names` in the directory `path_to/yolov3/cfg/`, with objects names - each in new line
	person
	car
	cat
	dog

##### 3. Create file `obj.data` in the directory `path_to/yolov3/cfg/`, containing (where **classes = number of objects**):

  ```
  classes= 3
  train  = /home/cai/workspace/yolov3/obj_train.txt
  valid  = /home/cai/workspace/yolov3/obj_test.txt
  names = /home/cai/workspace/yolov3/cfg/obj.names
  backup = /home/cai/workspace/yolov3/backup/
  ```

##### 4. Put image-files (.jpg) of your objects in the directory `path_to/yolov3/JPEGImages `

##### 5. You should label each object on images from your dataset: [[LabelImg]](https://github.com/tzutalin/labelImg) is a graphical image annotation tool

It will create `.txt`-file for each `.jpg`-image-file - in the same directory and with the same name, but with `.txt`-extension, and put to file: object number and object coordinates on this image, for each object in new line: 

`<object-class> <x_center> <y_center> <width> <height>`

  Where: 
  
  * `<object-class>` - integer object number from `0` to `(classes-1)`
  * `<x_center> <y_center> <width> <height>` - float values **relative** to width and height of image, it can be equal from `(0.0 to 1.0]`
  * for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
  * atention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner)

  For example for `img1.jpg` you will be created `img1.txt` containing:

  ```
  1 0.716797 0.395833 0.216406 0.147222
  0 0.687109 0.379167 0.255469 0.158333
  1 0.420312 0.395833 0.140625 0.166667
  ```

##### 6. Create file `obj_train.txt` & `obj_test.txt` in directory `path_to/yolov3/`, with filenames of your images, each filename in new line,for example containing:

  ```
  path_to/yolov3/JPEGImages/img1.jpg
  path_to/yolov3/JPEGImages/img2.jpg
  path_to/yolov3/JPEGImages/img3.jpg
  ```


##### 7. Download pre-trained weights for the convolutional layers (154 MB): https://pjreddie.com/media/files/darknet53.conv.74 and put to the directory `path_to/darknet/`


	wget https://pjreddie.com/media/files/darknet53.conv.74
	

##### 8. Start training by using the command line:

	./darknet detector train [path to .data file] [path to .cfg file] [path to pre-taining weights-darknet53.conv.74]
	
	[visualization]
	./darknet detector train path_to/yolov3/cfg/obj.data path_to/yolov3/cfg/yolov3.cfg darknet53.conv.74 2>1 | tee visualization/train_yolov3.log

##### 9. Start testing by using the command line:
 
	./darknet detector test path_to/yolov3/cfg/obj.data path_to/yolov3/cfg/yolov3.cfg path_to/yolov3/backup/yolov3_final.weights path_to/yolov3/test/test_img.jpg


