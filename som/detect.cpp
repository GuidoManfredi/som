#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "opencv_display/LocaPose.h"

#include "sod.h"
#include "FilesManager.h"

using namespace std;
using namespace cv;
using namespace pcl;
namespace enc = sensor_msgs::image_encodings;

Sod detecter;
Object object;
Mat P = Mat::eye(4, 4, CV_32F);

Object loadObject (char* path);
opencv_display::LocaPose mat2msg (Mat P);

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

	//imshow ("Input", image);
    //waitKey(1);
    int start = cv::getTickCount();
    P = detecter.process (image, depth);
    P = P.inv(); // for display
    cout << P << endl;
    int end = cv::getTickCount();
    float time_period = 1 / cv::getTickFrequency();
    ROS_INFO("Procesing time: %f s.", (end - start) * time_period);
}
// ./bin/detect_model /camera/depth_registered/points pose purfruit
// rosrun opencv_display display_poses /camera/rgb/image_rect_color pose
int main (int argc, char** argv) {
	assert (argc == 4 && "Usage : detect_objects in_registered_cloud_topic out_pose_topic object_name");
	ros::init(argc, argv, "som");
	ros::NodeHandle n;
	ros::Subscriber cloud_subscriber = n.subscribe(argv[1], 1, cloud_callback);
	ros::Publisher pose_publisher = n.advertise<opencv_display::LocaPose>(argv[2], 1);

    Object object = loadObject (argv[3]);
    detecter.setObject (object);
	while (ros::ok()) {
        ros::spinOnce();

        opencv_display::LocaPose msg;
        msg = mat2msg (P);
   	    pose_publisher.publish(msg);
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

opencv_display::LocaPose mat2msg (Mat P) {
    opencv_display::LocaPose msg;
    msg.t00 = P.at<float>(0, 0);    msg.t01 = P.at<float>(0, 1);    msg.t02 = P.at<float>(0, 2);    msg.t03 = P.at<float>(0, 3);
    msg.t10 = P.at<float>(1, 0);    msg.t11 = P.at<float>(1, 1);    msg.t12 = P.at<float>(1, 2);    msg.t13 = P.at<float>(1, 3);
    msg.t20 = P.at<float>(2, 0);    msg.t21 = P.at<float>(2, 1);    msg.t22 = P.at<float>(2, 2);    msg.t23 = P.at<float>(2, 3);
    return msg;
}
