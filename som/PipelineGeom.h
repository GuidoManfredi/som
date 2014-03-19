#pragma once
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>

#include "Object.h"
#include "Calibrate.h"

class PipelineGeom
{
  public:
    PipelineGeom ();
    void process (Object object, View current);
    float computeTrainPoseNotCalibrated (View current, View &train, std::vector<cv::DMatch> matches);
    float computeTrainPose (View current, View &train, std::vector<cv::DMatch> matches);
    float computeCurrentPose (View &current, View train, std::vector<cv::DMatch> matches);

  private:
    void object2viewFrame (cv::Mat points3d, cv::Mat &Rvo, cv::Mat &tvo);
    void solveCalibration (cv::Mat points3d, cv::Mat points2d,
                            cv::Mat &K, cv::Mat &R, cv::Mat &t);
    void mySolvePnP (cv::Mat points3d, cv::Mat points2d, cv::Mat K,
                     cv::Mat &Rov, cv::Mat &tov,
                     cv::Mat &p3d_inliers, cv::Mat &p2d_inliers);

    void transform (cv::Mat points3d, cv::Mat R, cv::Mat t,
                    cv::Mat &points3d_out);
    void getBarycentre (cv::Mat points3d, cv::Mat &t);
    void pointsFromIndex (cv::Mat p3d, cv::Mat p2d, std::vector<int> inliers,
                          cv::Mat &p3d_inliers, cv::Mat &p2d_inliers);
};
