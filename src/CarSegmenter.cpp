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

vector<vector<Point>> CarSegmenter::findContoursSimple(const Mat& img) {
    vector<vector<Point>> contours;
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    return contours;
}

void CarSegmenter::drawContourSimple(Mat& img, const vector<vector<Point>>& contours) {
    for (const auto& contour : contours) {
        drawContours(img, vector<vector<Point>>{contour}, -1, Scalar(0, 255, 0), 2);
    }
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

vector<RotatedRect> CarSegmenter::filterBBoxes(const vector<RotatedRect>& bboxes) {
    const double minSize = 1000;
    const double maxDist = 50;
    const float minOverlap = 0.01f;
    
    // Use a flag in the pair: true for originally large boxes, false for small ones.
    vector<pair<RotatedRect, bool>> largeBoxes;
    vector<pair<RotatedRect, bool>> smallBoxes;
    
    // Separate bounding boxes into large (true) and small (false)
    for (const auto& box : bboxes) {
        if (box.size.area() >= minSize)
            largeBoxes.push_back({box, true});
        else
            smallBoxes.push_back({box, false});
    }
    
    // Iterative merge for large boxes among themselves (phase 1).
    bool mergedLarge = true;
    while (mergedLarge) {
        mergedLarge = false;
        for (size_t i = 0; i < largeBoxes.size(); i++) {
            for (size_t j = i + 1; j < largeBoxes.size(); j++) {
                RotatedRect& box1 = largeBoxes[i].first;
                RotatedRect& box2 = largeBoxes[j].first;
                if (norm(box1.center - box2.center) < maxDist) {
                    float intersectionArea = rotatedRectIntersectionArea(box1, box2);
                    float unionArea = box1.size.area() + box2.size.area() - intersectionArea;
                    float overlapRatio = intersectionArea / unionArea;
                    if (overlapRatio >= minOverlap) {
                        // Merge box1 and box2 into a new rotated rectangle.
                        vector<Point2f> points(8);
                        Point2f pts1[4], pts2[4];
                        box1.points(pts1);
                        box2.points(pts2);
                        for (int k = 0; k < 4; k++) {
                            points[k] = pts1[k];
                            points[k + 4] = pts2[k];
                        }
                        box1 = minAreaRect(points);
                        // Keep the flag as true for large boxes.
                        largeBoxes.erase(largeBoxes.begin() + j);
                        mergedLarge = true;
                        break;
                    }
                }
            }
            if (mergedLarge)
                break;
        }
    }
    
    // Phase 2: Merge small boxes (flag false) with large boxes only if the large box’s flag is false.
    // This avoids merging between large boxes.
    for (const auto& smallBox : smallBoxes) {
        bool merged = false;
        // Only merge a small box with a large box that wasn’t originally large.
        for (auto& largeBox : largeBoxes) {
            if (largeBox.second == false) {
                if (norm(smallBox.first.center - largeBox.first.center) < maxDist) {
                    double intersectionArea = rotatedRectIntersectionArea(smallBox.first, largeBox.first);
                    double unionArea = smallBox.first.size.area() + largeBox.first.size.area() - intersectionArea;
                    double overlapRatio = intersectionArea / unionArea;
                    vector<Point2f> points(8);
                    Point2f pts1[4], pts2[4];
                    smallBox.first.points(pts1);
                    largeBox.first.points(pts2);
                    for (int k = 0; k < 4; k++) {
                        points[k] = pts1[k];
                        points[k + 4] = pts2[k];
                    }
                    largeBox.first = minAreaRect(points);
                    merged = true;
                    break;
                }
            }
        }
        // If the small box was not merged, you could choose to add it.
        // Uncomment the line below if you need to include unmerged small boxes.
        // if (!merged) largeBoxes.push_back(smallBox);
    }
    
    // Extract the RotatedRect objects from the pairs.
    vector<RotatedRect> result;
    for (const auto& item : largeBoxes) {
        result.push_back(item.first);
    }
    
    return result;
}

void CarSegmenter::drawBBoxes(Mat& img, const vector<RotatedRect>& bboxes) {
    for (const auto& bbox : bboxes) {
        Point2f vertices[4];
        bbox.points(vertices);
        for (int i = 0; i < 4; i++) line(img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
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

bool CarSegmenter::isCarInParking(const RotatedRect& carBox, const vector<RotatedRect>& parkingSpots) {
    Point2f center = carBox.center;
    const double tolerance = 5.0; // tolerance in pixels

    for (const auto &spot : parkingSpots) {
        Point2f vertices[4];
        spot.points(vertices);
        vector<Point2f> poly(vertices, vertices + 4);
        double distance = pointPolygonTest(poly, center, true);
        // Accept if within the tolerance (distance can be negative if slightly outside)
        if (distance >= -tolerance)
            return true;
    }
    return false;
}

Mat CarSegmenter::segmentCar (const vector<RotatedRect>& bboxes, const vector<RotatedRect>& groundtruthBBoxes, const Mat& mask, Mat& img) {
   // For each bbox take the correspondence portion of the mask and apply it to the image
    for (const auto& bbox: bboxes) {
        Point2f vertices[4];
        bbox.points(vertices);

        Mat carMask = Mat::zeros(mask.size(), CV_8UC1);
        vector<Point> carContour(vertices, vertices + 4);
        fillPoly(carMask, vector<vector<Point>>{carContour}, Scalar(255));

        Mat carSegment;
        bitwise_and(mask, mask, carSegment, carMask);

        Mat colorMask;
        cvtColor(carSegment, colorMask, COLOR_GRAY2BGR);

        Scalar color = isCarInParking(bbox, groundtruthBBoxes) ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        colorMask.setTo(color, carSegment > 0);

        img.setTo(color, carSegment > 0);
    }


    return img;
}