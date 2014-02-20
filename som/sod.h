#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Object.h"
#include "Pipeline2D.h"
#include "PipelineGeom.h"

class Sod
{
  public:
    Sod ();
    void setObject (Object object);
    cv::Mat process (const cv::Mat image, const cv::Mat depth);
    cv::Mat process (View &current);

  private:
    void match (View current, std::vector<View> trains, std::vector<std::vector<cv::DMatch> > &matches);
    void match (View current, View train, std::vector<cv::DMatch> &matches);
    View createView (cv::Mat image, cv::Mat depth);

    Object object_;

    Pipeline2D pipeline2d_;
    PipelineGeom pipelineGeom_;
};

