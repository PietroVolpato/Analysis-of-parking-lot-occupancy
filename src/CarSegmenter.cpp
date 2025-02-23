// PIETRO VOLPATO

#include "CarSegmenter.h"

using namespace cv;
using namespace std;

//------------------------------
// Gamma Correction Function
//------------------------------
// Performs gamma correction on the input image using a lookup table.
Mat CarSegmenter::gammaCorrection(const Mat& img, const double gamma) {
    // Create a lookup table with 256 entries (for each possible pixel value)
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i) {
        // Compute the gamma-corrected value
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    Mat correctedImg = img.clone();
    // Apply the lookup table to perform gamma correction
    LUT(img, lookUpTable, correctedImg);

    return correctedImg;
}

//------------------------------
// Convert to Grayscale Function
//------------------------------
// Converts a color image to grayscale.
Mat CarSegmenter::convertToGrayscale(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    return gray;
}

//------------------------------
// Equalization Function
//------------------------------
// Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast.
Mat CarSegmenter::equalization(const Mat& img) {
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    Mat equalizedImg;
    clahe->apply(img, equalizedImg);
    return equalizedImg;
}

//------------------------------
// Apply ROI Function
//------------------------------
// Applies a Region Of Interest (ROI) mask to the image by masking out a specified polygon.
Mat CarSegmenter::applyRoi(const Mat& img) {
    // Create a white mask (all pixels allowed)
    Mat mask = Mat::ones(img.size(), CV_8UC1) * 255;
    
    // Define the ROI polygon to remove (mask out).
    // The polygon is defined relative to the image dimensions.
    vector<Point> roi = {
        Point(static_cast<int>(img.cols * 0.65), 0),
        Point(img.cols, 0),
        Point(img.cols, static_cast<int>(img.rows * 0.4)),
        Point(static_cast<int>(img.cols * 0.65), 0)
    };
    
    // Fill the defined ROI with black (i.e., mask out this region)
    fillPoly(mask, vector<vector<Point>>{roi}, Scalar(0));
    
    Mat masked;
    // If the input image is colored, convert the mask to 3 channels and apply it.
    if (img.channels() == 3) {
        Mat maskColor;
        cvtColor(mask, maskColor, COLOR_GRAY2BGR);
        bitwise_and(img, maskColor, masked);
    } else {
        bitwise_and(img, mask, masked);
    }
    return masked;
}

//------------------------------
// Preprocess Image Function
//------------------------------
// Converts the image to grayscale, applies the specified enhancement (gamma correction or equalization),
// and then applies the ROI.
Mat CarSegmenter::preprocessImage(const Mat& img, String type) {
    Mat gray = convertToGrayscale(img);
    if (type == "gamma")
        return applyRoi(gammaCorrection(gray, 0.5));
    else if (type == "equalize")
        return applyRoi(equalization(gray));
    else
        return applyRoi(gray);
}

//------------------------------
// Find Contours (Simple)
//------------------------------
// Finds external contours in the binary image.
vector<vector<Point>> CarSegmenter::findContoursSimple(const Mat& img) {
    vector<vector<Point>> contours;
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    return contours;
}

//------------------------------
// Draw Contours (Simple)
//------------------------------
// Draws the provided contours on the image.
void CarSegmenter::drawContourSimple(Mat& img, const vector<vector<Point>>& contours) {
    for (const auto& contour : contours) {
        drawContours(img, vector<vector<Point>>{contour}, -1, Scalar(0, 255, 0), 2);
    }
}

//------------------------------
// Find Bounding Boxes from Contours
//------------------------------
// For each contour, computes the rotated bounding box (minAreaRect).
vector<RotatedRect> CarSegmenter::findBBoxes(const vector<vector<Point>>& contours) {
    vector<RotatedRect> bboxes;
    for (const auto& contour : contours) {
        RotatedRect bbox = minAreaRect(contour);
        bboxes.push_back(bbox);
    }
    return bboxes;
}

//------------------------------
// Intersection Area of RotatedRect
//------------------------------
// Computes the intersection area between two rotated rectangles.
float CarSegmenter::rotatedRectIntersectionArea(const RotatedRect& rect1, const RotatedRect& rect2) {
    vector<Point2f> intersection_pts;
    int result = rotatedRectangleIntersection(rect1, rect2, intersection_pts);
    if (result == INTERSECT_FULL || result == INTERSECT_PARTIAL) {
        return (float)contourArea(intersection_pts);
    }
    return 0.0;
}

//------------------------------
// Filter Bounding Boxes
//------------------------------
// Filters and merges bounding boxes based on their size, distance, and overlap criteria.
// Boxes with area >= minSize are considered large.
vector<RotatedRect> CarSegmenter::filterBBoxes(const vector<RotatedRect>& bboxes) {
    const double minSize = 1000;
    const double maxDist = 50;
    const float minOverlap = 0.01f;
    
    // Separate bounding boxes into large (flag true) and small (flag false)
    vector<pair<RotatedRect, bool>> largeBoxes;
    vector<pair<RotatedRect, bool>> smallBoxes;
    
    for (const auto& box : bboxes) {
        if (box.size.area() >= minSize)
            largeBoxes.push_back({box, true});
        else
            smallBoxes.push_back({box, false});
    }
    
    // Phase 1: Iteratively merge large boxes among themselves
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
                        // Merge box1 and box2 by computing the minimum area rectangle that contains both
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
    
    // Phase 2: Merge small boxes with large boxes if they are close and overlapping
    for (const auto& smallBox : smallBoxes) {
        bool merged = false;
        // Only merge a small box with a large box that wasn't originally large
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
    }
    
    // Extract the merged bounding boxes
    vector<RotatedRect> result;
    for (const auto& item : largeBoxes) {
        result.push_back(item.first);
    }
    
    return result;
}

//------------------------------
// Draw Bounding Boxes
//------------------------------
// Draws rotated bounding boxes on the image.
void CarSegmenter::drawBBoxes(Mat& img, const vector<RotatedRect>& bboxes) {
    for (const auto& bbox : bboxes) {
        Point2f vertices[4];
        bbox.points(vertices);
        for (int i = 0; i < 4; i++)
            line(img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
    }
}

//------------------------------
// Display Image
//------------------------------
// Displays an image in a window.
void CarSegmenter::showImages(const Mat& img) {
    imshow("Image", img);
    waitKey(0);
}

//------------------------------
// Train Background Subtractor
//------------------------------
// Trains the background model using a set of training images.
void CarSegmenter::trainBg(const vector<Mat>& trainingVector) {
    for (const auto& img : trainingVector) {
        Mat temp_mask;
        bg->apply(img, temp_mask, 0.01); // Use a small learning rate for gradual adaptation.
    }
}

//------------------------------
// Apply Background Subtractor
//------------------------------
// Applies the trained background subtractor to an image to obtain the foreground mask.
Mat CarSegmenter::applyBg(const Mat& img) {
    Mat fgMask;
    bg->apply(img, fgMask, 0); // Use learning rate 0 so the model remains unchanged.
    return fgMask;
}

//------------------------------
// Enhance Mask
//------------------------------
// Enhances the binary mask by applying morphological operations and thresholding.
Mat CarSegmenter::enhanceMask(const Mat& mask) {
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    threshold(mask, mask, 200, 255, THRESH_BINARY);
    return mask;
}

//------------------------------
// Check if Car is in Parking Spot
//------------------------------
// Determines if a car (represented by a rotated bounding box) is inside any of the parking spots.
// The check is done by testing whether the center of the car is inside the parking spot polygon.
bool CarSegmenter::isCarInParking(const RotatedRect& carBox, const vector<RotatedRect>& parkingSpots) {
    Point2f center = carBox.center;
    const double tolerance = 5.0; // Tolerance in pixels

    for (const auto &spot : parkingSpots) {
        Point2f vertices[4];
        spot.points(vertices);
        vector<Point2f> poly(vertices, vertices + 4);
        double distance = pointPolygonTest(poly, center, true);
        // Accept if the center is within the tolerance (distance can be negative if slightly outside)
        if (distance >= -tolerance)
            return true;
    }
    return false;
}

//------------------------------
// Segment Car
//------------------------------
// For each detected car bounding box, extract the corresponding region from the mask,
// color it red if the car is in a parking spot, or green otherwise, and apply it to the original image.
Mat CarSegmenter::segmentCar(const vector<RotatedRect>& bboxes, const vector<RotatedRect>& groundtruthBBoxes, const Mat& mask, Mat& img) {
    for (const auto& bbox: bboxes) {
        Point2f vertices[4];
        bbox.points(vertices);

        // Create a mask for the car using the rotated rectangle
        Mat carMask = Mat::zeros(mask.size(), CV_8UC1);
        vector<Point> carContour(vertices, vertices + 4);
        fillPoly(carMask, vector<vector<Point>>{carContour}, Scalar(255));

        // Extract the car segment from the mask
        Mat carSegment;
        bitwise_and(mask, mask, carSegment, carMask);

        // Convert the segment to a 3-channel image
        Mat colorMask;
        cvtColor(carSegment, colorMask, COLOR_GRAY2BGR);

        // Choose color: red if in parking spot, green otherwise
        Scalar color = isCarInParking(bbox, groundtruthBBoxes) ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        // Color the car segment and apply it to the original image
        colorMask.setTo(color, carSegment);
        img.setTo(color, carSegment);
    }
    return img;
}

//------------------------------
// New Method Using Connected Components
//------------------------------
// Uses connectedComponentsWithStats to find blobs in the mask,
// draws the blobs on the image with a specified color,
// and returns the bounding boxes as standard Rect objects.
vector<Rect> CarSegmenter::newmethod(Mat& mask, Mat& img) {
    Mat labels, stats, centroids;
    // Label connected components in the mask
    int numComponents = connectedComponentsWithStats(mask, labels, stats, centroids);

    int minArea = 500;   // Minimum area for a "car"
    int maxArea = 10000; // Maximum plausible area

    vector<Rect> bboxes;

    for (int i = 0; i < numComponents; i++) {
        // Get bounding box stats for this component
        int left   = stats.at<int>(i, CC_STAT_LEFT);
        int top    = stats.at<int>(i, CC_STAT_TOP);
        int width  = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);

        Rect bbox(left, top, width, height);
        if (bbox.area() > minArea && bbox.area() < maxArea) 
            bboxes.push_back(bbox);
        
    }
    return bboxes;
}

//------------------------------
// Create Segmentation Mask
//------------------------------
// Creates a segmentation mask from the image based on specific color values.
// If a pixel is exactly (0,0,255) [red], the mask value is set to 1; otherwise, it is set to 2, as requested in the assignment.
Mat CarSegmenter::createSegmentMask(const Mat& img) {
    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3b pixel = img.at<Vec3b>(i, j);
            if (pixel == Vec3b(0, 0, 255))
                mask.at<uchar>(i, j) = 1;
            else if (pixel == Vec3b(0, 255, 0))
                mask.at<uchar>(i, j) = 2;
        }
    }
    return mask;
}
