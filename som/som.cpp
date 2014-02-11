#include "som.h"

using namespace std;
using namespace cv;

Som::Som() {}

void Som::setIntrinsic (Mat K) {
    K.copyTo(K_);
}

void Som::loadIntrinsic (string calibration_file) {
    Mat K(3,3,CV_32F);
    FileStorage r_fs;
    r_fs.open (calibration_file, cv::FileStorage::READ);
    r_fs["camera_matrix"]>>K;
    r_fs.release ();
    setIntrinsic (K);
}

Object Som::model (std::vector<cv::Mat> images) {
    assert (K_.data && "K not set.");

    add_views (images);
    return create_object ();
}

Object Som::create_object () {
    Object object;
    object.views_ = views_;
    return object;
}

View Som::create_view (Mat K, Mat image) {
    View view;
    pipeline2d_.getGray (image, view.view_);
    pipeline2d_.extractFeatures (view.view_, view.keypoints_, view.descriptors_);
    view.K_ = K;
    return view;
}

void Som::add_views (vector<Mat> images) {
    for (size_t i = 0; i < images.size(); ++i) {
        add_view (images[i]);
    }
}

void Som::add_view (Mat image) {
    View v = create_view (K_, image);
    views_.push_back (v);
}
