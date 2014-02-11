#pragma once

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

class View
{
  public:
    View ();
    void points3dFromMatches (std::vector<cv::DMatch> matches,
                             cv::Mat &points3d);
    void points2dFromMatches (std::vector<cv::DMatch> matches,
                             cv::Mat &points2d);
    double reprojectionError (const cv::Mat Rov, const cv::Mat tov, const cv::Mat points3d, const cv::Mat points2d);
    double reprojectionError (const cv::Mat points3d, const cv::Mat points2d);

    void write (cv::FileStorage& fs) const;
    void read (const cv::FileNode& node);

    cv::Mat K_;

    cv::Mat view_;
    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;

    cv::Mat points3d_;
    cv::Mat Rov_;
    cv::Mat tov_;

    bool initialised;
};

//These write and read functions must exist as per the inline functions in operations.hpp
static void write(cv::FileStorage& fs, const std::string&, const View& x){
  x.write(fs);
}
static void read(const cv::FileNode& node, View& x, const View& default_value = View()){
  if(node.empty())
    x = default_value;
  else
    x.read(node);
}
