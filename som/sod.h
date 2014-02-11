#pragma once

#include <opencv2/core/core.hpp>

#include "Object.h"
#include "Pipeline2D.h"
#include "PipelineGeom.h"

class Sod
{
  public:
    Sod ();
    void setObject (Object object);
    void process (cv::Mat image, cv::Mat depth);
    void process (View current);

  private:
    void match (View current, std::vector<View> trains, std::vector<std::vector<cv::DMatch> > &matches);
    void match (View current, View train, std::vector<cv::DMatch> matches);
    View createView (cv::Mat image, cv::Mat depth);

    Object object_;

    Pipeline2D pipeline2d_;
    PipelineGeom pipelineGeom_;
};

