# Darknet and Yolo for ROS
Darknet/yolo wrapped in a ros-package for running Darknet/Yolo with ROS. 
All credit goes to Joseph Chet Redmon - you might enjoy his [resume](https://pjreddie.com/static/Redmon%20Resume.pdf) - and co. for implementing a deep learning framework [Darknet](http://pjreddie.com/darknet/) and the super fast object detector [YOLO](https://arxiv.org/abs/1506.02640) and [YOLO9000](https://pjreddie.com/media/files/papers/YOLO9000.pdf).

The make-file have been converted to a CMakeLists.
(Code test on ubuntu 16.04, cuda 8, opencv 3.2.0-dev)

### Install ROS and make workspace
Follow the ROS [installation guide](http://wiki.ros.org/ROS/Installation) and [tutorials]().

### Place package in your workspace src-folder
Go to ros-workspace source folder in a terminal

	cd [workspace]/src/

Clone folder in terminal.

	$ git clone https://github.com/PeteHeine/DarknetROS

### Build project (GPU)
#### For CPU 
Open [workspace]/src/DarknetROS/CMakeList.txt and ensure that

	set(GPU 0)

#### For GPU
Software requires CUDA to be installed. 
Go to [download page](https://developer.nvidia.com/cuda-downloads) select your platform, download and follow instructions.

Open [workspace]/src/DarknetROS/CMakeList.txt and ensure that

	set(GPU 1)
#### Build

Go to workspace and run catkin make. 

	cd [workspace]
	catkin_make


### Test with ROS and usb cam
Clone usb_cam ros package to workspace.
Go again to ros-workspace source folder.

	cd [workspace]/src/

Clone folder

	git clone https://github.com/bosch-ros-pkg/usb_cam.git

Build packages

	cd [workspace]
	catkin_make

Source ros-workspace

	source devel/setup.bash

Find the pre-trained model on the following [page](https://pjreddie.com/darknet/yolo/) or simply use a [direct link](http://pjreddie.com/media/files/yolo.weights). For a smaller model use [direct link](http://pjreddie.com/media/files/tiny-yolo-voc.weights)

Move file to weight-folder

	cd [workspace]/src/DarknetROS
	(mkdir weights)

	# Normal model
	cp ~/Downloads/yolo.weights ./weights 

	# Fast model
	cp ~/Downloads/tiny-yolo-voc.weights ./weights/


Run launch file 

	roslaunch darknet_ros RunWebCamAndYolo.launch	



