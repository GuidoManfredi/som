#include "som.h"
#include "FilesManager.h"

using namespace std;
using namespace cv;

Object model (string dir);

Som modeler;

// ./bin/create_model /home/gmanfred/devel/datasets/my_objects/purfruit

int main (int argc, char** argv) {
    Object object = model (argv[1]);
    return 0;
}

Object model (string dir) {
    cout << "Loading images" << endl;
    FilesManager fm;
    vector<Mat> images = fm.getImages (dir);

    cout << "Modeling" << endl;
    modeler.loadIntrinsic ("/home/gmanfred/.ros/camera_info/my_xtion.yml");
    Object object = modeler.model (images);

    cout << "Saving object" << endl;
    string object_name = fm.getDirName (dir);
    string object_filename = object_name + ".yaml";
    FileStorage fs(object_filename, FileStorage::WRITE);
    fs << object_name << object;
    fs.release();

    return object;
}
