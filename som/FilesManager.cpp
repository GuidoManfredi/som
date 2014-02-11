#include "FilesManager.h"

using namespace std;
using namespace boost::filesystem;
using namespace cv;

FilesManager::FilesManager () {}

vector<Mat> FilesManager::getImages (string images_path) {
    vector<Mat> images;
    boost::filesystem::path bf_images_path(images_path);
    if ( is_regular_file(bf_images_path) ) {
        Mat image = getImage (bf_images_path.string());
        images.push_back (image);
    }
    else if ( is_directory(bf_images_path) ) {
        directory_iterator end_itr;
        for ( directory_iterator itr(bf_images_path); itr != end_itr; ++itr ) {
            Mat image = getImage (itr->path().string());
            if (image.data) {
                images.push_back (image);
            }
        }
    }
    return images;
}
////////////////////////////////////////////////////////////////////////////////
//  PRIVATE METHODS
////////////////////////////////////////////////////////////////////////////////
Mat FilesManager::getImage (string image_path) {
    Mat image;
    boost::filesystem::path bf_images_path(image_path);
    if (extension(bf_images_path) == ".png"
        || extension(bf_images_path) == ".jpg") {
        image = imread (bf_images_path.string(), CV_LOAD_IMAGE_COLOR);
        if(! image.data )
            cout << "Could not open or find the image at " << bf_images_path.string() << endl;
    } else {
        cout << "Did not read " << bf_images_path.string() << endl;
    }
    return image;
}

string FilesManager::getDirName (string train_dir) {
    boost::filesystem::path bf_images_path(train_dir);
    return bf_images_path.stem().string();
}
