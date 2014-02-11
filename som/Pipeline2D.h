#pragma once

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <GL/glew.h>
#include <GL/glut.h>
#include "/home/gmanfred/devel/sandbox/SiftGPU/src/SiftGPU/SiftGPU.h"

typedef std::pair<int,double> IndexDistance;

class Pipeline2D {
  public:
    Pipeline2D ();
    // Various
    void getGray(const cv::Mat& image, cv::Mat& gray);
    int filterNaNKeyPoints (cv::Mat depth, std::vector<cv::KeyPoint> keypoints,
                              std::vector<int> &filtered_kpts_index,
                              std::vector<cv::KeyPoint> &filtered_kpts,
                              std::vector<cv::Point3f> &filtered_p3d);
    // Features functions
    bool extractKeypoints(const cv::Mat& image,
						std::vector<cv::KeyPoint>& keypoints);
    bool extractDescriptors(const cv::Mat image, std::vector<cv::KeyPoint> keypoints,
						  cv::Mat& descriptors);
    bool extractDescriptors(const cv::Mat image, std::vector<int> keypoints_index,
                          cv::Mat& descriptors);
    bool extractFeatures(const cv::Mat& image,
  						 std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    // Match
	void matchNoMinimum (const cv::Mat &desc1, const cv::Mat &desc2,
                         std::vector<cv::DMatch>& matches);
	bool match(const cv::Mat &desc1, const cv::Mat &desc2,
               std::vector<cv::DMatch>& matches);

    unsigned int minNumberMatchesAllowed_;

  private:
    struct IndexDistanceComparatorClass {
        bool operator() (const IndexDistance& l, const IndexDistance& r) {
            //std::cout << l.second << " " << r.second << std::endl;
            return l.second < r.second;
        }
    } IndexDistanceComparator;

    cv::KeyPoint GPUkpt2kpt (SiftGPU::SiftKeypoint key);
    cv::Mat GPUdesc2desc (std::vector<float> descriptors);
    cv::Mat GPUdesc2desc (std::vector<float> descriptors, std::vector<int> kpts_index);
    std::vector<float> desc2GPUdesc (cv::Mat descs);

    SiftGPU sift_;
};
