#include "CarSegmenter.h"

using namespace cv;
using namespace std;

Mat CarSegmenter::gammaCorrection (const Mat& img, const double gamma) {
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    Mat correctedImg = img.clone();
    LUT(img, lookUpTable, correctedImg);

    return correctedImg;
}

Mat CarSegmenter::convertToGrayscale (const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    return gray;
}

Mat CarSegmenter::equalization (const Mat& img) {
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    Mat equalizedImg;
    clahe->apply(img, equalizedImg);

    return equalizedImg;
}

vector<Mat> CarSegmenter::loadImages(const int sequence) {
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

vector<RotatedRect> CarSegmenter::getBBoxes(String filePath) {
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
        return {
            RotatedRect(Point2f(0, 0), Size2f(0, 0), 0)
        };
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

        // Compute the rotated bounding box and store it in the vector
        boundingBoxes.push_back(minAreaRect(points));
    }

    return boundingBoxes;
}

Mat CarSegmenter::preprocessImage (const Mat& img, String type) {
    Mat gray = convertToGrayscale(img);
    if (type == "gamma") return gammaCorrection(gray, 0.5);
    else if (type == "equalize") return equalization(gray);
    else return gray;
}

pair<vector<vector<Point>>, vector<Vec4i>> CarSegmenter::findContoursImg(const Mat& img) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    return {contours, hierarchy};
}

vector<RotatedRect> CarSegmenter::findBBoxes(const vector<vector<Point>>& contours) {
    vector<RotatedRect> bboxes;
    for (const auto& contour : contours) {
        RotatedRect bbox = minAreaRect(contour);
        bboxes.push_back(bbox);
    }

    return bboxes;
}

float CarSegmenter::rotatedRectIntersectionArea(const RotatedRect& rect1, const RotatedRect& rect2) {
    vector<Point2f> intersection_pts;
    int result = rotatedRectangleIntersection(rect1, rect2, intersection_pts);
    if (result == INTERSECT_FULL || result == INTERSECT_PARTIAL) {
        return (float)contourArea(intersection_pts);
    }
    return 0.0;
}

vector<RotatedRect> CarSegmenter::mergeRotatedBBoxes(const vector<RotatedRect>& inputBoxes) {
    const float overlapThreshold = 0.3;
    vector<RotatedRect> mergedBoxes;
    vector<bool> merged(inputBoxes.size(), false);

    for (size_t i = 0; i < inputBoxes.size(); i++) {
        if (merged[i]) continue; // Already merged
        RotatedRect mergedRect = inputBoxes[i];

        for (size_t j = i + 1; j < inputBoxes.size(); j++) {
            if (merged[j]) continue;

            float interArea = rotatedRectIntersectionArea(mergedRect, inputBoxes[j]);
            float area1 = mergedRect.size.area();
            float area2 = inputBoxes[j].size.area();

            float overlap = interArea / min(area1, area2); // Percentage of overlap
            float dist = norm(mergedRect.center - inputBoxes[j].center);

            if (overlap > overlapThreshold || dist < 50) { // Merge if there is overlap or if centers are close
                mergedRect = RotatedRect(
                    (mergedRect.center + inputBoxes[j].center) * 0.5, // New average center
                    Size2f(max(mergedRect.size.width, inputBoxes[j].size.width),   // Maximum width
                           max(mergedRect.size.height, inputBoxes[j].size.height)),  // Maximum height
                    (mergedRect.angle + inputBoxes[j].angle) / 2 // Average of angles
                );
                merged[j] = true; // Mark as merged
            }
        }

        mergedBoxes.push_back(mergedRect);
    }

    return mergedBoxes;
}

vector<RotatedRect> CarSegmenter::filterBBoxes(const vector<RotatedRect>& bboxes) {
    vector<RotatedRect> filtered;
    for (const auto& box : bboxes) {
        double area = box.size.area();
        if (area > 500) {
            // To avoid division by zero, check height
            if (box.size.height != 0) {
                double ratio = box.size.width / box.size.height;
                if (ratio > 1.0 && ratio < 3.5) {
                    filtered.push_back(box);
                }
            }
        }
    }
    return filtered;

}

void CarSegmenter::drawContoursImg(Mat& img, const vector<vector<Point>>& contours, const vector<Vec4i>& hierarchy) {
    for (size_t i = 0; i < contours.size(); i++) {
        if (hierarchy[i][3] == -1) {
            drawContours(img, contours, i, Scalar(0, 255, 0), 2, LINE_8, hierarchy);
        }
    }
}

void CarSegmenter::drawBBoxes(Mat& img, const vector<RotatedRect>& bboxes) {
    for (const auto& bbox : bboxes) {
        Point2f vertices[4];
        bbox.points(vertices);
        for (int i = 0; i < 4; i++) {
            line(img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
        }
    }
}

void CarSegmenter::showImages(const Mat& img) {
    imshow("Image", img);
    waitKey(0);
}

void CarSegmenter::trainBg (const vector<Mat>& trainingVector) {
    for (const auto& img : trainingVector) {
        Mat temp_mask;
        bg->apply(img, temp_mask, 0.01);
    }
}

Mat CarSegmenter::applyBg (const Mat& img) {
    Mat fgMask;
    bg->apply(img, fgMask, 0);

    return fgMask;
}

Mat CarSegmenter::enhanceMask(const Mat& mask) {
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    threshold(mask, mask, 200, 255, THRESH_BINARY);

    return mask;
}

Mat CarSegmenter::segmentCar (const vector<RotatedRect>& bboxes, const Mat& mask, const Mat& img) {
   // For each bbox take the correspondence portion of the mask and apply it to the image
    Mat segmentedImg = Mat::zeros(img.size(), img.type());
    

    return segmentedImg;
}