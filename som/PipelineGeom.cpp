#include "PipelineGeom.h"

#include "opencv2/core/core_c.h" // fore cv::reduce

using namespace std;
using namespace cv;

PipelineGeom::PipelineGeom () {}

double PipelineGeom::computeTrainPose (View current, View &train, vector<DMatch> matches) {
    if ( matches.size() < 4) {
        cout << "ComputeTrainPose : not enought matches (" << matches.size() << ")." << endl;
        return -1;
    }

    Mat points2d, kinect_points3d;
    train.points2dFromMatches (matches, points2d);
    current.points3dFromMatches (matches, kinect_points3d);

    Mat Rov, tov;
    object2viewFrame (kinect_points3d,
                      Rov, tov);
    // transform 3d points from camera to object object frame
    Mat object_points3d;
    Mat Rvo = Rov.t();
    Mat tvo = -tov;
    transform (kinect_points3d, Rvo, tvo,
               object_points3d);
    //cout << object_points3d << endl;
    //cout << points2d << endl;
    mySolvePnP (object_points3d, points2d, train.K_, train.Rov_, train.tov_);

    double reprojection_error = train.reprojectionError (object_points3d, points2d);
    return reprojection_error;
    cout << "OpenCV reprojection error train pose: " << reprojection_error << endl;
}

double PipelineGeom::computeCurrentPose (View &current, View train, vector<DMatch> matches) {
    Mat points2d, camera_points3d, world_points3d;
    train.points2dFromMatches (matches, points2d);
    current.points3dFromMatches (matches, camera_points3d);

    Mat RR, tt;
    mySolvePnP (camera_points3d, points2d, train.K_, RR, tt);
    // convert A = P * (P')-1 to P' with P' = A-1 * P
    Mat PP = Rt2P (RR, tt);
    Mat P = Rt2P (train.Rov_, train.tov_);
    Mat P2 = PP.inv() * P;
    P2Rt (P2, current.Rov_, current.tov_);

    double reprojection_error = current.reprojectionError (RR, tt, camera_points3d, points2d);
    return reprojection_error;
}
////////////////////////////////////////////////////////////////////////////////
//  PRIVATE METHODS
////////////////////////////////////////////////////////////////////////////////
void PipelineGeom::object2viewFrame (Mat points3d, Mat &Rov, Mat &tov) {
    // compute object to points frame transform
    tov = Mat::zeros(1, 3, CV_64F);
    reduce (points3d, tov, 0, CV_REDUCE_AVG);
    Rov = Mat::eye (3, 3, CV_64F); // keep same rotation as current camera
}

void PipelineGeom::mySolvePnP (Mat p3d, Mat p2d, Mat K, Mat &Rov, Mat &tov) {
    Mat rvec = Mat::zeros (1, 3, CV_64F);
    tov = Mat::zeros (1, 3, CV_64F);

    solvePnPRansac (p3d, p2d, K, vector<double>(), rvec, tov);

    Rodrigues (rvec, Rov);
}

void PipelineGeom::P2Rt (Mat P, Mat &R, Mat &t) {
    R.at<double>(0,0) = P.at<double>(0,0);
    R.at<double>(0,1) = P.at<double>(0,1);
    R.at<double>(0,2) = P.at<double>(0,2);
    t.at<double>(0) = P.at<double>(0, 3);

    R.at<double>(1,0) = P.at<double>(1,0);
    R.at<double>(1,1) = P.at<double>(1,1);
    R.at<double>(1,2) = P.at<double>(1,2);
    t.at<double>(1) = P.at<double>(1,3);

    R.at<double>(2,0) = P.at<double>(2,0);
    R.at<double>(2,1) = P.at<double>(2,1);
    R.at<double>(2,2) = P.at<double>(2,2);
    t.at<double>(2) = P.at<double>(2,3);
}

Mat PipelineGeom::Rt2P (Mat R, Mat t) {
    Mat P = Mat::zeros(4, 4, CV_64F);
    P.at<double>(0,0) = R.at<double>(0,0);
    P.at<double>(0,1) = R.at<double>(0,1);
    P.at<double>(0,2) = R.at<double>(0,2);
    P.at<double>(0,3) = t.at<double>(0);

    P.at<double>(1,0) = R.at<double>(1,0);
    P.at<double>(1,1) = R.at<double>(1,1);
    P.at<double>(1,2) = R.at<double>(1,2);
    P.at<double>(1,3) = t.at<double>(1);

    P.at<double>(2,0) = R.at<double>(2,0);
    P.at<double>(2,1) = R.at<double>(2,1);
    P.at<double>(2,2) = R.at<double>(2,2);
    P.at<double>(2,3) = t.at<double>(2);

    P.at<double>(3,3) = 1;

    return P;
}

void PipelineGeom::transform (cv::Mat points3d, cv::Mat R, cv::Mat t,
                cv::Mat &points3d_out) {
    int n = points3d.rows;
    points3d_out = Mat (n, 3, CV_64F);
    for ( int i = 0; i < n; ++i ) {
        // produit de transposé : (R * pt.t() + t.t()).t() == pt * R.t() + t
        Mat res = points3d.row(i) * R.t() + t;
        res.copyTo(points3d_out.row(i));
    }
}

void PipelineGeom::getBarycentre (Mat points3d,
                                  Mat &t) {
    t = Mat::zeros (1, 3, CV_64F);
    double n = points3d.rows;
    for ( int i = 0; i < n; ++i ) {
        t += points3d.row(i);
    }
    t /= n;
}
