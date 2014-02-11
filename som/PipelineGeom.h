#pragma once
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>

#include "Object.h"

class PipelineGeom
{
  public:
    PipelineGeom ();
    void process (Object object, View current);

    double computeTrainPose (View current, View &train, std::vector<cv::DMatch> matches);
    double computeCurrentPose (View &current, View train, std::vector<cv::DMatch> matches);

  private:
    void object2viewFrame (cv::Mat points3d, cv::Mat &Rvo, cv::Mat &tvo);
    void mySolvePnP (cv::Mat points3d, cv::Mat points2d, cv::Mat K, cv::Mat &Rov, cv::Mat &tov);
    void P2Rt (cv::Mat P, cv::Mat &R, cv::Mat &t);
    cv::Mat Rt2P (cv::Mat R, cv::Mat t);
    void transform (cv::Mat points3d, cv::Mat R, cv::Mat t,
                    cv::Mat &points3d_out);
    void getBarycentre (cv::Mat points3d, cv::Mat &t);
};
