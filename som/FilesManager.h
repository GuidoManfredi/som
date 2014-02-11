#pragma once

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class FilesManager
{
  public:
    FilesManager ();
    std::vector<cv::Mat> getImages (std::string images_path);
    std::string getDirName (std::string train_dir);
  private:
    cv::Mat getImage (std::string image_path);
};
