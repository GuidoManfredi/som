#pragma once

#include <iostream>

#include "View.h"

class Object
{
  public:
    Object ();
    void save (std::string filepath);
    void write (cv::FileStorage& fs) const;
    void read (const cv::FileNode& node);

    std::vector<View> views_;
};

static void write(cv::FileStorage& fs, const std::string&, const Object& x){
  x.write(fs);
}
static void read(const cv::FileNode& node, Object& x, const Object& default_value = Object()){
  if(node.empty())
    x = default_value;
  else
    x.read(node);
}


