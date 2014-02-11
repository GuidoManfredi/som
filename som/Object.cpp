#include "Object.h"

#include <fstream>

using namespace std;
using namespace cv;

Object::Object () {}

void Object::write(cv::FileStorage& fs) const {
    cout << "kikou write object" << endl;
    int number_views = views_.size();
    fs << "{" << "number_views" << number_views;
    for ( int i = 0; i < number_views; ++i ) {
        ostringstream view_stream;
        view_stream << "view_" << i;
        fs << view_stream.str() << views_[i];
    }
    fs << "}";
}

void Object::read(const FileNode& node) {
    cout << "kikou read object" << endl;
    int number_views = 0;
    node["number_views"] >> number_views;
    views_.resize(number_views);
    for ( int i = 0; i < number_views; ++i ) {
        ostringstream view_stream;
        view_stream << "view_" << i;
        node[view_stream.str()] >> views_[i];
    }
}
