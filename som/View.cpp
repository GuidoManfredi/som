#include "View.h"

using namespace std;
using namespace cv;

View::View () {
    initialised = false;
    //extractFeatures (image, keypoints_, descriptors_);
}

void View::points3dFromMatches (vector<DMatch> matches,
                                Mat &points3d) {
    points3d = Mat (matches.size(), 3, CV_32F);
    for ( size_t i = 0; i < matches.size(); ++i ) {
        int idx = matches[i].queryIdx;
//        cout << points3d_.at<Point3f>(idx) << endl;
        points3d.at<Point3f>(i) = points3d_.at<Point3f>(idx);
//        cout << points3d.at<Point3f>(i) << endl;
    }
}

void View::points2dFromMatches (vector<DMatch> matches,
                                Mat &points2d) {
    points2d = Mat (matches.size(), 2, CV_32F);
    for ( size_t i = 0; i < matches.size(); ++i ) {
        int idx = matches[i].trainIdx;
        points2d.at<Point2f>(i) = keypoints_[idx].pt;
    }
}

float View::reprojectionError (const Mat points3d, const Mat points2d) {
    return reprojectionError (Rov_, tov_, points3d, points2d);
}

float View::reprojectionError (const Mat Rov, const Mat tov, const Mat points3d, const Mat points2d) {
    return reprojectionError (Rov, tov, K_, points3d, points2d);
}

float View::reprojectionError (const Mat Rov, const Mat tov, const Mat K, const Mat points3d, const Mat points2d) {
    float total_error = 0;
    Mat rvec;
    Rodrigues (Rov, rvec);
    Mat tvec = tov;

    int n = points3d.rows;
    for ( int i = 0; i < n; ++i ) {
        Mat projected_points2d (1, 2, CV_32F);
        projectPoints (points3d.row(i), rvec, tvec, K, vector<float>(), projected_points2d);
        projected_points2d = projected_points2d.reshape (1, 1);
        total_error += norm(projected_points2d, points2d.row(i), NORM_L2);
    }

    return total_error / n;
}

Mat View::transform2Object () {
    Mat P = Mat::eye (4, 4, CV_32F);
    if ( !Rov_.data || !tov_.data )
        return P;

    Rov_ = Rov_.t(); // TODO comprendre pourquoi !!!
    P.at<float>(0, 0) = Rov_.at<float>(0, 0); P.at<float>(0, 1) = Rov_.at<float>(0, 1); P.at<float>(0, 2) = Rov_.at<float>(0, 2);
    P.at<float>(1, 0) = Rov_.at<float>(1, 0); P.at<float>(1, 1) = Rov_.at<float>(1, 1); P.at<float>(1, 2) = Rov_.at<float>(1, 2);
    P.at<float>(2, 0) = Rov_.at<float>(2, 0); P.at<float>(2, 1) = Rov_.at<float>(2, 1); P.at<float>(2, 2) = Rov_.at<float>(2, 2);

    tov_ = Rov_ * tov_.t();
    P.at<float>(0, 3) = tov_.at<float>(0); P.at<float>(1, 3) = tov_.at<float>(1); P.at<float>(2, 3) = tov_.at<float>(2);
    return P;
}
////////////////////////////////////////////////////////////////////////////////
//  STORAGE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
void View::write(cv::FileStorage& fs) const {
    cout << "Writing view" << endl;
    fs << "{" << "Intrinsic" << K_
              << "Rotation" << Rov_
              << "Translation" << tov_
              << "keypoints" << keypoints_
              << "descriptors" << descriptors_
              << "}";
}

void View::read(const FileNode& node) {
    cout << "Reading view" << endl;
    node["Intrinsic"] >> K_;
    node["Rotation"] >> Rov_;
    node["Translation"] >> tov_;
    FileNode keypoints = node["keypoints"];
    cv::read(keypoints, keypoints_);
    node["descriptors"] >> descriptors_;
}
