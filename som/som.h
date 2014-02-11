#pragma once

#include "Object.h"
#include "Pipeline2D.h"

class Som
{
  public:
    Som ();
    void setIntrinsic (cv::Mat K);
    void loadIntrinsic (std::string calibration_file);
    Object model (std::vector<cv::Mat> images);

  private:
    Object create_object ();
    View create_view (cv::Mat K, cv::Mat image);
    void add_views (std::vector<cv::Mat> images);
    void add_view (cv::Mat image);

    Pipeline2D pipeline2d_;
    cv::Mat K_;
    std::vector<View> views_;
};
