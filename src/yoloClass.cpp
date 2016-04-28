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
#include <htf_safe_msgs/SAFEObstacleMsg.h>
#include <sensor_msgs/image_encodings.h>
//#include <sensor_msgs/CompressedImage.h>
#include <image_transport/image_transport.h>
//#include <camera_info_manager/camera_info_manager.h>
#include <math.h>


extern "C" {
#include "box.h"
#include "yoloInterface.h"
#include "image.h"
#include "utils.h"
}

using namespace std;
using namespace cv;

char *voc_names2[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor","barrel","birdnest"};
#define NCLASSES 22
class MyNode {
public:
	MyNode() :
		nh("~"), it(nh) {
			nh.param<std::string>("model_cfg",model_cfg,"/cfg/yoloSmall20.cfg");
			nh.param<std::string>("weightfile",weightfile,"/weight/yoloSmall20.weights");
			nh.param<std::string>("topic_name",topic_name,"/usb_cam/image_raw");
			nh.param<float>("threshold",threshold,0.2);
			
			// Distance estimate.
			nh.param<double>("FOV_verticalDeg",FOV_verticalDeg,47.0);
			nh.param<double>("FOV_horizontalDeg",FOV_horizontalDeg,83.0);
			nh.param<double>("angleTiltDegrees",angleTiltDegrees,7.0);
			nh.param<double>("cameraHeight",cameraHeight,1.9);

			FOV_verticalRad = FOV_verticalDeg*M_PI/180;
			FOV_horizontalRad = FOV_horizontalDeg*M_PI/180;
			angleTiltRad = angleTiltDegrees*M_PI/180;
			//cinfor_ = boost::shared_ptr<camera_info_manager::CameraInfoManager>(new camera_info_manager::CameraInfoManager(nh, "test", ""));
			//sub_image = it.subscribeCamera(topic_name.c_str(), 1, &MyNode::onImage, this);
			sub_image = it.subscribe(topic_name.c_str(), 1, &MyNode::onImage, this);
			pub_image = it.advertise("imageYolo", 1);
			pub_bb = nh.advertise<std_msgs::Float64MultiArray>("BBox", 1);
			pub_bbSAFE = nh.advertise<htf_safe_msgs::SAFEObstacleMsg>("BBoxSAFE", 1);
			test = 1;
			maxDetections = load_yolo_model((char*)model_cfg.c_str(), (char*)weightfile.c_str());
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TOPIC: %s",topic_name.c_str());
			boxes = (box*)calloc(maxDetections, sizeof(box));
			probs = (float**)calloc(maxDetections, sizeof(float *));
			for(int j = 0; j < maxDetections; ++j) probs[j] = (float*)calloc(NCLASSES, sizeof(float *));
		};

	~MyNode() {
		free(boxes);
		free(probs);
		for(int j = 0; j < maxDetections; ++j) free(probs[j]);
	}
	;

	//void onImage(const sensor_msgs::ImageConstPtr& msg,const sensor_msgs::CameraInfoConstPtr& p) {
	void onImage(const sensor_msgs::ImageConstPtr& msg) {
		
		if(test==1)
		{
			test = 0;
			 
			cv_bridge::CvImagePtr cv_ptr;
			try {
				cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
			} catch (cv_bridge::Exception& e) {
				ROS_ERROR("cv_bridge exception: %s", e.what());
				return;
			}
			//imageResize =1;
			//resize(cv_ptr->image, img, Size(), imageResize, imageResize);
			// Convert to Darknet image format.
			image im = OpencvMat2DarkNetImage(cv_ptr->image);
			// Yolo detections are returned as boxes and probs.
			execute_yolo_model(im, threshold,boxes, probs); // Returns bounding boxes and probabilities.
			publish_detections(cv_ptr->image, maxDetections, threshold, boxes, probs,voc_names2);
			
			free_image(im);
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
	Mat publish_detections(Mat img, int num, float thresh, box *boxesIn, float **probsIn, char **names)
	{
		int i;
		int cDetections = 0;
		box_prob* detections = (box_prob*)calloc(maxDetections, sizeof(box_prob));
		for(i = 0; i < num; ++i){
			int topClass = max_index(probs[i],NCLASSES);
			float prob = probs[i][topClass];
			if(prob > thresh){
				int width = pow(prob, 1./2.)*10+1; // line thickness
				//printf("%s: %.2f\n", names[topClass], prob);
				box b = boxes[i];
				 Scalar useColor(0, 0, 0);
				float x  = (b.x-b.w/2.)*(float)(img.cols);
				float y = (b.y-b.h/2.)*(float)(img.rows);
				float w   = b.w*(float)(img.cols);
				float h   = b.h*(float)(img.rows);
				printf("bb: %f %f %f %f \n", x,y,w,h);
				rectangle(img, Rect(x,y,w,h), useColor, 2, 8, 0);
				char numstr[30];
				sprintf(numstr, "%s %.2f",names[topClass], prob); 
				putText(img, numstr, Point(x+4,y-14+h),FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, 8, false);

				// Detection in x and y coordinates (with x, y as upper left corner)
				detections[cDetections].x = x;
				detections[cDetections].y = y;
				detections[cDetections].w = w;
				detections[cDetections].h = h;
				detections[cDetections].prob = prob;
				detections[cDetections].objectType = topClass;

				cDetections++;
			}
		}


		/* Creating visual marker
		visualization_msgs::Marker marker;
		marker.header.frame_id = "/laser";
		marker.header.stamp = ros::Time();
		marker.ns = "my_namespace";
		marker.id = 0;
		marker.type = visualization_msgs::Marker::CYLINDER;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;*/

		
		// An estimate of the distance to the object is calculated using the camera setup.
		// Estimate is based on two assumptions: 1) The surface is flat. 2) The bottom of the bounding box is the bottom of the detected object.
		if(1){
			printf("Start bboxSAFE \n");
			double resolutionVertical = img.rows;
			double resolutionHorisontal = img.cols;
			htf_safe_msgs::SAFEObstacleMsg msgObstacle;
			// MAYBE CLEARING IS NEEDED
			msgObstacle.xCoordinate.clear();
			msgObstacle.yCoordinate.clear();
			msgObstacle.zCoordinate.clear();
			msgObstacle.quality.clear();
			msgObstacle.objectType.clear();
			msgObstacle.objectID.clear();

			msgObstacle.header.stamp = ros::Time::now();

			for (int n = 0; n < cDetections;n++){
				double buttomRowPosition = detections[n].y+detections[n].h; // bbs(n,2)-bbs(n,4);
				double ColPosition = detections[n].x+detections[n].w/2; // bbs(n,2)-bbs(n,4);

				double distance = tan(M_PI/2-(angleTiltRad+FOV_verticalRad/2) + FOV_verticalRad*(resolutionVertical-buttomRowPosition)/resolutionVertical)*cameraHeight;
				double angle =((ColPosition-resolutionHorisontal/2)/resolutionHorisontal)*FOV_verticalRad;
				double xCoordinate = cos(angle)*distance;
				double yCoordinate = sin(angle)*distance;
				msgObstacle.xCoordinate.push_back(xCoordinate);
				msgObstacle.yCoordinate.push_back(yCoordinate);
				msgObstacle.zCoordinate.push_back(0.0);
				msgObstacle.quality.push_back(detections[n].prob);
				msgObstacle.objectType.push_back(detections[n].objectType);
				msgObstacle.objectID.push_back(0);
				//cout << "x1:" << bbs[n].x1 << ", y2:" << bbs[n].y2 << ", w3:" << bbs[n].width3 << ", h4:" << bbs[n].height4 << ", s5: " << bbs[n].score5 << ",a5: " << bbs[n].angle << endl;
				//cout << "Distance: " <<  bbs[n].distance << endl;
			}
			pub_bbSAFE.publish(msgObstacle);
		}


		// Create bounding box publisher (multi array)
		std_msgs::Float64MultiArray bboxMsg;
		bboxMsg.data.clear();

		for (int iBbs = 0; iBbs < cDetections; ++iBbs) {
//			bboxMsg.data.push_back(bbs[iBbs].distance);
//			bboxMsg.data.push_back(bbs[iBbs].angle);
			bboxMsg.data.push_back(detections[iBbs].x/img.cols);
			bboxMsg.data.push_back(detections[iBbs].y/img.rows);
			bboxMsg.data.push_back(detections[iBbs].w/img.cols);
			bboxMsg.data.push_back(detections[iBbs].h/img.rows);
			bboxMsg.data.push_back(detections[iBbs].prob);
			bboxMsg.data.push_back(detections[iBbs].objectType);
		}
		pub_bb.publish(bboxMsg);


		// Create image publisher showing yolo detections.
		//sensor_msgs::CameraInfoPtr cc(new sensor_msgs::CameraInfo(cinfor_->getCameraInfo()));
		sensor_msgs::ImagePtr msg_out = cv_bridge::CvImage(std_msgs::Header(),"bgr8", img).toImageMsg();
		msg_out->header.stamp = ros::Time::now();
		//pub_image.publish(msg_out, cc);
		pub_image.publish(msg_out);
		//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
		//imshow( "Display window", img);
		free (detections);
		return img;
	}
private:
	double imageResize;
	float threshold;
	std::string model_cfg;
	std::string weightfile;
	std::string topic_name;
	cv::Mat img;
	ros::NodeHandle nh;
	image_transport::ImageTransport it;
	//image_transport::CameraPublisher pub_image;
	//image_transport::CameraSubscriber sub_image;
	image_transport::Publisher pub_image;
	image_transport::Subscriber sub_image;
	ros::Publisher pub_bb;
	ros::Publisher pub_bbSAFE;
	//boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfor_;
	bool test;
	box *boxes;
	float **probs;
	int maxDetections;
	double FOV_verticalDeg,FOV_horizontalDeg,angleTiltDegrees,cameraHeight;
	double FOV_verticalRad, FOV_horizontalRad,angleTiltRad;
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
