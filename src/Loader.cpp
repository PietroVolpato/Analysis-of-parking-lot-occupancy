#include "Loader.h"

using namespace cv;
using namespace std;

//------------------------------
// Load images from sequence
//------------------------------
// Loads images from a specific sequence directory based on the sequence number.
vector<Mat> Loader::loadImagesFromSequence (const int sequence) {
    vector<String> fileNames;
    string basePath;
    switch(sequence) {
        case 0: basePath = "../data/sequence0/frames"; break;
        case 1: basePath = "../data/sequence1/frames"; break;
        case 2: basePath = "../data/sequence2/frames"; break;
        case 3: basePath = "../data/sequence3/frames"; break;
        case 4: basePath = "../data/sequence4/frames"; break;
        case 5: basePath = "../data/sequence5/frames"; break;
        default:
            cerr << "Invalid sequence number: " << sequence << endl;
            return {};
    }
    glob(basePath, fileNames);

    vector<Mat> imgs;
    for (const auto& file : fileNames) {
        Mat img = imread(file);
        if (img.empty()) {
            cerr << "Error loading image: " << file << endl;
            continue;
        }
        imgs.push_back(img);
    }
    return imgs;
}

//------------------------------
// Load images from path
//------------------------------
// Loads images from a specific directory based on the path.
vector<Mat> Loader::loadImagesFromPath (String path) {
    vector<String> fileNames;
    glob(path, fileNames);

    vector<Mat> imgs;
    for (const auto& file : fileNames) {
        Mat img = imread(file);
        if (img.empty()) {
            cerr << "Error loading image: " << file << endl;
            continue;
        }
        imgs.push_back(img);
    }
    return imgs;
}

//------------------------------
// Get Bounding Boxes from XML
//------------------------------
// Loads parking space bounding boxes from an XML file using pugixml.
vector<RotatedRect> Loader::getBBoxes(String filePath) {
    // Load the XML file
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filePath.c_str());
    if (!result) {
        cout << "Error loading XML file." << endl;
        return {};
    }

    // Get the root element "parking"
    pugi::xml_node parking = doc.child("parking");
    if (!parking) {
        cout << "Root element not found." << endl;
        return { RotatedRect(Point2f(0, 0), Size2f(0, 0), 0) };
    }

    vector<RotatedRect> boundingBoxes;  // Vector to store bounding boxes

    // Iterate over each <space> element
    for (pugi::xml_node space = parking.child("space"); space; space = space.next_sibling("space"))
    {
        // Find the <contour> element containing the points
        pugi::xml_node contour = space.child("contour");
        if (!contour)
            continue;

        // Extract all points from the contour
        vector<Point> points;
        for (pugi::xml_node point = contour.child("point"); point; point = point.next_sibling("point"))
        {
            int x = point.attribute("x").as_int();
            int y = point.attribute("y").as_int();
            points.push_back(Point(x, y));
        }

        if (points.empty())
            continue;

        // Compute the rotated bounding box (minAreaRect) and store it
        boundingBoxes.push_back(minAreaRect(points));
    }

    return boundingBoxes;
}

//------------------------------
// Load mask from sequence
//------------------------------
// Loads masks from a specific sequence directory based on the sequence number.
vector<Mat> Loader::loadMask (const int sequence) {
    vector<String> fileNames;
    string basePath;
    switch(sequence) {
        case 1: basePath = "../data/sequence1/masks"; break;
        case 2: basePath = "../data/sequence2/masks"; break;
        case 3: basePath = "../data/sequence3/masks"; break;
        case 4: basePath = "../data/sequence4/masks"; break;
        case 5: basePath = "../data/sequence5/masks"; break;
        default:
            cerr << "Invalid sequence number: " << sequence << endl;
            return {};
    }
    glob(basePath, fileNames);

    vector<Mat> masks;
    for (const auto& file : fileNames) {
        Mat mask = imread(file, IMREAD_GRAYSCALE);
        if (mask.empty()) {
            cerr << "Error loading mask: " << file << endl;
            continue;
        }
        masks.push_back(mask);
    }
    return masks;
}