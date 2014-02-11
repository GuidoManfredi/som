#include "View.h"

using namespace std;
using namespace cv;

View::View () {
    //extractFeatures (image, keypoints_, descriptors_);
}

void View::points3dFromMatches (vector<DMatch> matches,
                                Mat &points3d) {
    points3d = Mat (matches.size(), 3, CV_64F);
    for ( size_t i = 0; i < matches.size(); ++i ) {
        int idx = matches[i].queryIdx;
        points3d.at<Point3d>(i) = points3d_.at<Point3d>(idx);
    }
}

void View::points2dFromMatches (vector<DMatch> matches,
                                Mat &points2d) {
    points2d = Mat (matches.size(), 2, CV_64F);
    for ( size_t i = 0; i < matches.size(); ++i ) {
        int idx = matches[i].trainIdx;
        points2d.at<Point2d>(i) = keypoints_[idx].pt;
    }
}

double View::reprojectionError (const Mat points3d, const Mat points2d) {
    return reprojectionError (Rov_, tov_, points3d, points2d);
}

double View::reprojectionError (const Mat Rov, const Mat tov, const Mat points3d, const Mat points2d) {
    double total_error = 0;
    Mat rvec;
    Rodrigues (Rov, rvec);
    Mat tvec = tov;

    int n = points3d.rows;
    for ( int i = 0; i < n; ++i ) {
        Mat projected_points2d (1, 2, CV_64F);
        projectPoints (points3d.row(i), rvec, tvec, K_, vector<double>(), projected_points2d);
        projected_points2d = projected_points2d.reshape (1, 1);
        total_error += norm(projected_points2d, points2d.row(i), NORM_L2);
    }

    return total_error / n;
}
////////////////////////////////////////////////////////////////////////////////
//  STORAGE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
void View::write(cv::FileStorage& fs) const {
    cout << "kikou write view" << endl;
    fs << "{" << "keypoints" << keypoints_
              << "descriptors" << descriptors_
              << "Rotation" << Rov_
              << "Translation" << tov_ << "}";
}

void View::read(const FileNode& node) {
    cout << "kikou read view" << endl;
    FileNode keypoints = node["keypoints"];
    cv::read(keypoints, keypoints_);
    node["descriptors"] >> descriptors_;
    node["Rotation"] >> Rov_;
    node["Translation"] >> tov_;
}
