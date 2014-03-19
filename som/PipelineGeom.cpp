#include "PipelineGeom.h"

#include "opencv2/core/core_c.h" // fore cv::reduce

using namespace std;
using namespace cv;

PipelineGeom::PipelineGeom () {}

float PipelineGeom::computeTrainPoseNotCalibrated (View current, View &train, vector<DMatch> matches) {
    if ( matches.size() < 4) {
        cout << "ComputeTrainPoseNotCalibrated : not enought matches (" << matches.size() << ")." << endl;
        return -1;
    }
    Mat points2d, kinect_points3d;
    train.points2dFromMatches (matches, points2d);
    current.points3dFromMatches (matches, kinect_points3d);
    //cout << "Pose 2d: " << points2d.size() << endl;
    //cout << "Pose 3d: " << kinect_points3d.size() << endl;
    Mat Rov, tov;
    object2viewFrame (kinect_points3d,
                      Rov, tov);
    // transform 3d points from camera to object object frame
    Mat object_points3d;
    Mat Pov = Rt2P(Rov, tov);
    Mat Pvo = Pov.inv();
    Mat Rvo, tvo;
    P2Rt (Pvo, Rvo, tvo);
    transform (kinect_points3d, Rvo, tvo,
               object_points3d);
    //cout << object_points3d << endl;
    //cout << points2d << endl;
    Mat p3d_inliers, p2d_inliers;
    solveCalibration (object_points3d, points2d, train.K_, train.Rov_, train.tov_);
                        //,p3d_inliers, p2d_inliers);
    float reprojection_error = train.reprojectionError (p3d_inliers, p2d_inliers);
    cout << "Reprojection error train pose: " << reprojection_error << endl;
    return reprojection_error;
}

float PipelineGeom::computeTrainPose (View current, View &train, vector<DMatch> matches) {
    if ( matches.size() < 4) {
        cout << "ComputeTrainPose : not enought matches (" << matches.size() << ")." << endl;
        return -1;
    }
    Mat points2d, kinect_points3d;
    train.points2dFromMatches (matches, points2d);
    current.points3dFromMatches (matches, kinect_points3d);
    //cout << "Pose 2d: " << points2d.size() << endl;
    //cout << "Pose 3d: " << kinect_points3d.size() << endl;
    Mat Rov, tov;
    object2viewFrame (kinect_points3d,
                      Rov, tov);
    // transform 3d points from camera to object object frame
    Mat object_points3d;
    Mat Pov = Rt2P(Rov, tov);
    Mat Pvo = Pov.inv();
    Mat Rvo, tvo;
    P2Rt (Pvo, Rvo, tvo);
    transform (kinect_points3d, Rvo, tvo,
               object_points3d);
    //cout << object_points3d << endl;
    //cout << points2d << endl;
    Mat p3d_inliers, p2d_inliers;
    mySolvePnP (object_points3d, points2d, train.K_, train.Rov_, train.tov_,
                p3d_inliers, p2d_inliers);
    float reprojection_error = train.reprojectionError (p3d_inliers, p2d_inliers);
    cout << "Reprojection error train pose: " << reprojection_error << endl;
    return reprojection_error;
}

float PipelineGeom::computeCurrentPose (View &current, View train, vector<DMatch> matches) {
    if ( matches.size() < 4) {
        cout << "ComputeTrainPose : not enought matches (" << matches.size() << ")." << endl;
        return -1;
    }
    Mat points2d, camera_points3d;
    train.points2dFromMatches (matches, points2d);
    current.points3dFromMatches (matches, camera_points3d);

    Mat RR, tt;
    Mat p3d_inliers, p2d_inliers;
    mySolvePnP (camera_points3d, points2d, train.K_, RR, tt,
                p3d_inliers, p2d_inliers);
    // convert A = P * (P')-1 to P' with P' = A-1 * P
    Mat A = Rt2P (RR, tt);
    Mat P = Rt2P (train.Rov_, train.tov_);
    Mat P2 = A.inv() * P;
    P2Rt (P2, current.Rov_, current.tov_);
    //current.tov_ = current.Rov_ * current.tov_.t();

    float reprojection_error = current.reprojectionError (RR, tt, train.K_, p3d_inliers, p2d_inliers);
    cout << "Reprojection error current pose: " << reprojection_error << endl;
    return reprojection_error;
}
////////////////////////////////////////////////////////////////////////////////
//  PRIVATE METHODS
////////////////////////////////////////////////////////////////////////////////
void PipelineGeom::object2viewFrame (Mat points3d, Mat &Rov, Mat &tov) {
    // compute object to points frame transform
    tov = Mat::zeros(1, 3, CV_32F);
    reduce (points3d, tov, 0, CV_REDUCE_AVG);
    Rov = Mat::eye (3, 3, CV_32F); // keep same rotation as current camera
}

void PipelineGeom::solveCalibration (cv::Mat points3d, cv::Mat points2d,
                                        cv::Mat &K, cv::Mat &R, cv::Mat &t) {
    solveCalibrateLinear (points3d, points2d, K, R, t);
}

void PipelineGeom::mySolvePnP (Mat p3d, Mat p2d, Mat K,
                               Mat &Rov, Mat &tov,
                               Mat &p3d_inliers, Mat &p2d_inliers) {
    Mat rvec = Mat::zeros (1, 3, CV_32F);
    Rov = Mat::eye (3, 3, CV_32F);
    tov = Mat::zeros (1, 3, CV_32F);
//    cout << p3d << endl;
//    cout << p2d << endl;
    vector<int> inliers_idx;
    solvePnPRansac (p3d, p2d, K, vector<float>(), rvec, tov,
                    false, 1000, 4.0, p3d.rows*0.8, inliers_idx, CV_EPNP); // 80% inliers
    // Reffine with LM
    if ( inliers_idx.size() > 4 ) {
//    cout << inliers_idx.size() << endl;
//    cout << p3d_inliers.size() << endl;
//    cout << p2d_inliers.size() << endl;
        pointsFromIndex (p3d, p2d, inliers_idx, p3d_inliers, p2d_inliers);
        solvePnP (p3d_inliers, p2d_inliers, K, vector<float>(), rvec, tov, true);
    } else {
        p3d_inliers = p3d;
        p2d_inliers = p2d;
    }
//    cout << "Result" << endl;
//    cout << rvec << endl;
//    cout << tov << endl;
    Rodrigues (rvec, Rov);

    Rov.convertTo (Rov, CV_32F);
    tov.convertTo (tov, CV_32F);
}

void PipelineGeom::transform (cv::Mat points3d, cv::Mat R, cv::Mat t,
                cv::Mat &points3d_out) {
    int n = points3d.rows;
    points3d_out = Mat (n, 3, CV_32F);
    for ( int i = 0; i < n; ++i ) {
        // produit de transposé : (R * pt.t() + t.t()).t() == pt * R.t() + t
        Mat res = points3d.row(i) * R.t() + t;
        res.copyTo(points3d_out.row(i));
    }
}

void PipelineGeom::getBarycentre (Mat points3d,
                                  Mat &t) {
    t = Mat::zeros (1, 3, CV_32F);
    float n = points3d.rows;
    for ( int i = 0; i < n; ++i ) {
        t += points3d.row(i);
    }
    t /= n;
}

void PipelineGeom::pointsFromIndex (Mat p3d, Mat p2d, vector<int> inliers,
                                    Mat &p3d_inliers, Mat &p2d_inliers) {
    p3d_inliers = Mat::zeros (inliers.size(), 3, CV_32F);
    p2d_inliers = Mat::zeros (inliers.size(), 2, CV_32F);
    for ( size_t n = 0; n < inliers.size(); ++n ) {
        int idx = inliers[n];
        p3d_inliers.at<Point3f>(n) = p3d.at<Point3f>(idx);
        p2d_inliers.at<Point2f>(n) = p2d.at<Point2f>(idx);
    }
}
