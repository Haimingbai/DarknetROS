#include <stdio.h>
#include <cv.h>
#include <vector>
#include <highgui.h>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <std_msgs/MultiArrayLayout.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>



extern "C" {
#include "box.h"
#include "yolo.h"
#include "image.h"
}

using namespace std;
using namespace cv;

class MyNode {
public:
	MyNode() :
		nh("~"), it(nh) {
			nh.param<std::string>("model_cfg",model_cfg,"/home/repete/DeepLearningStuff/darknet/cfg/yolo-small.cfg");
			nh.param<std::string>("weightfile",weightfile,"/home/repete/DeepLearningStuff/darknet/weights/yolo-small.weights");
			nh.param<std::string>("topic_name",topic_name,"/usb_cam/image_raw");
			cam_sub = it.subscribeCamera(topic_name.c_str(), 1, &MyNode::onImage, this);
			test = 1;
			load_yolo_model((char*)model_cfg.c_str(), (char*)weightfile.c_str());
		}
	;

	~MyNode() {

	}
	;

	void onImage(const sensor_msgs::ImageConstPtr& msg,const sensor_msgs::CameraInfoConstPtr& p) {
		
		if(test==1)
		{
			test = 0;
			// do all the stuff here
			ROS_ERROR("GOT Image");
			//convert  image to opencv
			ROS_ERROR("ImageHasBeenReceived"); 
			cv_bridge::CvImagePtr cv_ptr;
			try {
				cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
			} catch (cv_bridge::Exception& e) {
				ROS_ERROR("cv_bridge exception: %s", e.what());
				return;
			}
			imageResize =1;
			resize(cv_ptr->image, img, Size(), imageResize, imageResize);
			float thresh = 0.2;
			execute_yolo_model(OpencvMat2DarkNetImage(img), thresh);
			printf("State2");
			test = 1;
		}
	}
	image OpencvMat2DarkNetImage(Mat src)
	{
		unsigned char *data = (unsigned char *)src.data;
		int h = src.rows;
		int w = src.cols;
		int c = src.channels();
		int step = src.step1();

		image out = make_image(w, h, c);
		int i, j, k, count=0;

		for(k= c-1; k >= 0; --k){
			for(i = 0; i < h; ++i){
				for(j = 0; j < w; ++j){
					out.data[count++] = data[i*step + j*c + k]/255.;
				}
			}
		}
		return out;
	}
private:
	double FOV_verticalDeg,FOV_horizontal,angleTiltDegrees,cameraHeight;
	double imageResize;
	std::string model_cfg;
	std::string weightfile;
	std::string topic_name;
	cv::Mat img;
	ros::NodeHandle nh;
	image_transport::ImageTransport it;
	image_transport::CameraPublisher cam_pub;
	image_transport::CameraSubscriber cam_sub;
	boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfor_;
	bool test;
};


int main(int argc, char** argv) {

	ros::init(argc, argv, "darknet_stuff");

	MyNode node;

	ros::spin();
}

/*int main()
{
	char cfgfile[] = "/home/repete/DeepLearningStuff/darknet/cfg/yolo-small.cfg";
	char weightfile[] = "/home/repete/DeepLearningStuff/darknet/weights/yolo-small.weights";
	char filename[] = "/home/repete/DeepLearningStuff/darknet/data/dog.jpg";

	printf("State0");
	float thresh = 0.2;
	load_yolo_model(cfgfile, weightfile);
	printf("State1");
	execute_yolo_model(filename, thresh);
	printf("State2");
	return 0;
}*/
