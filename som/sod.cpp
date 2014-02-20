#include "sod.h"

using namespace std;
using namespace cv;

Sod::Sod () {

}

void Sod::setObject (Object object) {
    object_ = object;
}

Mat Sod::process (const Mat image, const Mat depth) {
    View view = createView (image, depth);
    return process (view);
}

Mat Sod::process (View &current) {
    vector<vector<DMatch> > matches;
    View best_view;
    int best_index = -1;
    float min_reprojection_error = 1000;
    match (current, object_.views_, matches);
    for ( size_t i = 0; i < matches.size(); ++i ) {
        float reprojection_error = 0;
        View tmp_view = object_.views_[i];
        if ( !object_.views_[i].initialised ) {
            cout << "Completing train image." << endl;
            reprojection_error = pipelineGeom_.computeTrainPose (current, tmp_view, matches[i]);
        } else {
            cout << "Computing for current image." << endl;
            //imshow ("Online training view", current.view_);
            //waitKey(0);
            reprojection_error = pipelineGeom_.computeCurrentPose (current, tmp_view, matches[i]);
        }

        if ( reprojection_error < min_reprojection_error
            && reprojection_error > 0 ) {
            best_view = tmp_view;
            best_index = i;
            min_reprojection_error = reprojection_error;
        }
    }

    object_.views_[best_index] = best_view;
    if ( !object_.views_[best_index].initialised )
        object_.views_[best_index].initialised = true;

    cout << "Min reprojection error " << min_reprojection_error << endl;
    cout << "Best index " << best_index << endl;

    return current.transform2Object ();
}

void Sod::match (View current, vector<View> trains, vector<vector<DMatch> > &matches) {
    matches.clear ();
    for ( size_t i = 0; i < trains.size(); ++i ) {
        vector<DMatch> tmp_matches;
        match (current, trains[i], tmp_matches);
        matches.push_back (tmp_matches);
    }
}

void Sod::match (View current, View train, vector<DMatch> &matches) {
    matches.clear ();

    pipeline2d_.match (current.descriptors_, train.descriptors_, matches);
//    cout << current.descriptors_.size() << endl;
//    cout << train.descriptors_.size() << endl;
//    cout << "Matches : " << matches.size() << endl;
}

View Sod::createView (Mat image, Mat depth) {
    View view;
    pipeline2d_.getGray (image, view.view_);
    vector<KeyPoint> raw_keypoints;
    pipeline2d_.extractKeypoints (view.view_, raw_keypoints);
    // remove NaN of depth from keypoints
    vector<int> keypoints_index;
    vector<Point3f> points3d;
    pipeline2d_.filterNaNKeyPoints (depth, raw_keypoints,
                                    keypoints_index, view.keypoints_, points3d);
    Mat(points3d).copyTo(view.points3d_);
    pipeline2d_.extractDescriptors (view.view_, keypoints_index, view.descriptors_);

    return view;
}

