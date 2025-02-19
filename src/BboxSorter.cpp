#include "BboxSorter.h"

using namespace cv;
using namespace std;

// Constructor
BboxSorter::BboxSorter(vector<RotatedRect> rects) : rectangles(std::move(rects)) {}

// Find the four extreme points (bottom-left, top-left, bottom-right, top-right)
vector<Point2f> BboxSorter::findExtremePoints() {
    Point2f bottomLeft = {FLT_MAX, -FLT_MAX};
    Point2f topLeft = {FLT_MAX, FLT_MAX};
    Point2f bottomRight = {-FLT_MAX, -FLT_MAX};
    Point2f topRight = {-FLT_MAX, FLT_MAX};
    int bl = 0;
    int tl = 0;
    int br = 0;
    int tr = 0;
    int i =0;

    for (const auto& rect : rectangles) {
        Point2f center = rect.center;

        if (center.y > bottomLeft.y) {
            bottomLeft = center;
            bl =i;
        }
        if (center.x < topLeft.x) {
            topLeft = center;
            tl = i;
        }
        if (center.x > bottomRight.x ) {
            bottomRight = center;
            br = i;
        }
        if (center.y < topRight.y) {
            topRight = center;
            tr = i;
        }
        i++;
    }

    return {bottomLeft, topLeft, bottomRight, topRight};
}

// Compute the homography matrix using the given source and destination points
Mat BboxSorter::computeHomography(const vector<Point2f>& srcPoints) {
    vector<Point2f> dstPoints = {
        {500, 500},  // Bottom-left (mapped to new coordinate space)
        {50, 500},    // Top-left
        {500, 50},    // Bottom-right
        {70, 50}       // Top-right
    };

    return getPerspectiveTransform(srcPoints, dstPoints);
}

// Apply homography transformation to all bounding box centers
vector<Point2f> BboxSorter::applyHomography(const Mat& H) {
    vector<Point2f> transformedCenters;
    vector<Point2f> originalCenters;

    for (const auto& rect : rectangles) {
        originalCenters.push_back(rect.center);
    }

    perspectiveTransform(originalCenters, transformedCenters, H);
    return transformedCenters;
}

// Sort bounding boxes using homography-based transformed coordinates
vector<RotatedRect> BboxSorter::sort() {
    if (rectangles.empty()) return {};

    // Step 1: Find four extreme points
    vector<Point2f> srcPoints = findExtremePoints();

    // Step 2: Compute Homography Matrix
    Mat H = computeHomography(srcPoints);

    // Step 3: Apply Homography Transformation
    vector<Point2f> transformedCenters = applyHomography(H);
    drawTransformedCenters(transformedCenters);

    // Step 4: Sort based on new coordinates
    vector<int> indices(rectangles.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Sort indices based on transformed coordinates
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    if (a >= transformedCenters.size() || b >= transformedCenters.size()) {
        std::cerr << "Error: Index out of bounds (a=" << a << ", b=" << b << ")\n";
        return false;
    }

    if (transformedCenters[a].y == transformedCenters[b].y)
        return transformedCenters[a].x < transformedCenters[b].x;
    return transformedCenters[a].y < transformedCenters[b].y;
    });

    // Generate sorted rectangles using indices
    std::vector<cv::RotatedRect> sortedRects;
    for (int idx : indices) {

        sortedRects.push_back(rectangles[idx]);
    }

    return sortedRects;
}
void BboxSorter::drawTransformedCenters(const std::vector<cv::Point2f>& transformedCenters) {
    // Create a blank image (600x600, black background)
    cv::Mat img = cv::Mat::zeros(1000, 1000, CV_8UC3);

    // Draw each transformed center as a small circle
    for (const auto& point : transformedCenters) {
        cv::circle(img, point, 5, cv::Scalar(0, 255, 0), -1);  // Green dot
    }

    // Show the image
    cv::imshow("Transformed Centers", img);
    cv::waitKey(0);  // Wait for key press before closing
}
