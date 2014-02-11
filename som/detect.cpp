#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "sod.h"
#include "FilesManager.h"

using namespace std;
using namespace cv;
using namespace pcl;
namespace enc = sensor_msgs::image_encodings;

Sod detecter;
Object object;
Mat P (4, 4, CV_32F);

Object loadObject (char* path);

void cloud_callback(const sensor_msgs::PointCloud2& msg) {
	PointCloud<PointXYZRGB>::Ptr cloud (new PointCloud<PointXYZRGB>);
	fromROSMsg (msg, *cloud);

	unsigned int h = cloud->height;
	unsigned int w = cloud->width;
	Mat depth (h, w, CV_32FC3);
	Mat image (h, w, CV_8UC3);
	for (size_t y=0; y<h; ++y) {
		for (size_t x=0; x<w; ++x) {
			PointXYZRGB pt;
			pt = cloud->at(x,y);
			depth.at<Vec3f>(y, x)[0] = pt.x;
			depth.at<Vec3f>(y, x)[1] = pt.y;
			depth.at<Vec3f>(y, x)[2] = pt.z;
			int rgb = *reinterpret_cast<int*>(&pt.rgb);
            image.at<Vec3b>(y, x)[0] = (rgb & 0xff); // B
            image.at<Vec3b>(y, x)[1] = ((rgb >> 8) & 0xff); // G
            image.at<Vec3b>(y, x)[2] = ((rgb >> 16) & 0xff); // R
		}
	}
    int start = cv::getTickCount();
	detecter.process (image, depth);
    int end = cv::getTickCount();
    float time_period = 1/cv::getTickFrequency();
    ROS_INFO("Procesing time: %f s.", (end - start)*time_period);
}

// ./bin/detect_model /camera/depth_registered/points pose purfruit
int main (int argc, char** argv) {
	assert (argc == 4 && "Usage : detect_objects in_registered_cloud_topic out_pose_topic object_name");

	ros::init(argc, argv, "POD");
	ros::NodeHandle n;
	ros::Subscriber cloud_subscriber = n.subscribe(argv[1], 1, cloud_callback);
	//ros::Publisher pose_publisher = n.advertise<geometry_msgs::PoseStamped>(argv[2], 1);

    Object object = loadObject (argv[4]);
    detecter.setObject (object);

	while (ros::ok()) {
        ros::spinOnce();

   	    //pose_publisher.publish(pose);
    }

    return 0;
}

Object loadObject (char* path) {
    string object_name (path);
    string object_file = object_name + ".yaml";
    FileStorage fs(object_file, FileStorage::READ);
    Object object;
    fs[object_name] >> object;
    return object;
}

