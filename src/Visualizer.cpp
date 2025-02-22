#include "Visualizer.h"

using namespace std;
using namespace cv;

// Constructor definition
Visualizer::Visualizer(int width, int height, std::vector<cv::RotatedRect>& parkingSpaces) 
    : minimap_width(width), minimap_height(height), rectangles(parkingSpaces) {}



void Visualizer::drawParkingSpaces(Mat &image, const vector<bool> &occupancyStatus) {
    Point center;
    
    for (size_t i = 0; i < rectangles.size(); ++i) {
        Point2f vertices[4];
        rectangles[i].points(vertices);
        center = static_cast<Point>(rectangles[i].center);
        
        // Set color based on occupancy status (Red for occupied, Green for free)
        Scalar color = occupancyStatus[i] ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        for (int j = 0; j < 4; j++) {
            line(image, vertices[j], vertices[(j + 1) % 4], color, 2);
        }

       // Draw the index at the center
        string indexText = to_string(i);
        putText(image, indexText, center, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
    }
}

void Visualizer::drawRotatedRect(Mat& image, Point2f center, float angle, Scalar color) {

    
    RotatedRect rrect(center, Size2f(Visualizer::spaceWidth, Visualizer::spaceHeight), angle);
    Point2f vertices[4];
    rrect.points(vertices);

    // Convert to a polygon and fill it
    vector<Point> contour(vertices, vertices + 4);
    fillPoly(image, vector<vector<Point>>{contour}, color);

    // Draw border line
    for (int i = 0; i < 4; i++) {
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 0), 1);
    }
}

void Visualizer::drawParkingRow(Mat& image, const Mat &H, vector<bool>& occupancy, int& index) {
    
    // Get transformed centers
    vector<Point2f> transformedCenters = Visualizer::applyHomography(H);
    Visualizer::clusterParkingSpaces(transformedCenters);
    int angle;
    for (int i = 0; i < rectangles.size(); i++) {
        // assigning angle
        if (rectangles[i].size.width > rectangles[i].size.height)
            angle = 45;
        else
            angle = - 45;
        
        // Assign color based on occupancy vector
        bool occupied = occupancy[index++];
        Scalar color = occupied ? Scalar(0, 0, 255) : Scalar(255, 0, 0); // Red for occupied, Blue for free

        drawRotatedRect(image, transformedCenters[i], angle, color);
    }
}

Mat Visualizer::overlaySmallOnLarge(const Mat& parkingSapaceClassified, const Mat& visual2D) {

    // Clone the large image to keep the original unchanged
    Mat outputImage = parkingSapaceClassified.clone();

    // The size for the small image (1/4 width of the large image)
    int newWidth = parkingSapaceClassified.cols / 4;
    int newHeight = (visual2D.rows * newWidth) / visual2D.cols;  // Maintain aspect ratio

   // Resize the small image
   Mat resizedSmallImage;
   resize(visual2D, resizedSmallImage, cv::Size(newWidth, newHeight));

   // Define the ROI (bottom-left corner)
   Rect roi(0, outputImage.rows - newHeight, newWidth, newHeight);

   // Copy small image into the ROI
   resizedSmallImage.copyTo(outputImage(roi));

   return outputImage;
}



Mat Visualizer::createMockMinimap(vector<bool> &occupancy) {
    Mat minimap(Visualizer::minimap_height, Visualizer::minimap_width, CV_8UC3, Scalar(255, 255, 255));
    // Step 1: Find four extreme points
    vector<Point2f> srcPoints = findExtremePoints();

    // Step 2: Compute Homography Matrix
    Mat H = computeHomography(srcPoints);

    int index = 0; // Track occupancy vector position
    drawParkingRow(minimap, H, occupancy, index);
    imshow("minimap", minimap);
    waitKey(0);

    return minimap;
}

vector<Point2f> Visualizer::findExtremePoints() {
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
Mat Visualizer::computeHomography(const vector<Point2f>& srcPoints) {
    vector<Point2f> dstPoints = {
        {380, 280},  // Bottom-left (mapped to new coordinate space)
        {50, 280},    // Top-left
        {380, 50},    // Bottom-right
        {50, 50}       // Top-right
    };

    return getPerspectiveTransform(srcPoints, dstPoints);
}

// Apply homography transformation to all bounding box centers
vector<Point2f> Visualizer::applyHomography(const Mat& H) {
    vector<Point2f> transformedCenters;
    vector<Point2f> originalCenters;

    for (const auto& rect : rectangles) {
        originalCenters.push_back(rect.center);
    }

    perspectiveTransform(originalCenters, transformedCenters, H);
    return transformedCenters;
}

void Visualizer::clusterParkingSpaces(vector<Point2f> centers) {
    if (centers.empty()) return;

    int K = 4; // Number of clusters
    int N = centers.size();

    // Prepare data for k-means clustering
    cv::Mat data(N, 1, CV_32F);
    for (int i = 0; i < N; i++) {
        data.at<float>(i, 0) = centers[i].y;
    }

    // Run k-means clustering
    cv::Mat labels, centersMat;
    cv::kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1.0),
               10, cv::KMEANS_PP_CENTERS, centersMat);

    // Resize vector for 4 clusters
    clusteredCenters.resize(K);

    // Assign centers to clusters
    for (int i = 0; i < N; i++) {
        int clusterIdx = labels.at<int>(i, 0);
        clusteredCenters[clusterIdx].push_back(centers[i]);
    }

    // Sort clusters based on increasing y-values of their centers
    std::sort(clusteredCenters.begin(), clusteredCenters.end(),
              [](const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
                  return a.front().y < b.front().y;
              });

    // Print the clusters
    for (int i = 0; i < clusteredCenters.size(); i++) {
        std::cout << "Cluster " << i << ":\n";
        for (const auto& p : clusteredCenters[i]) {
            std::cout << "  Center: (" << p.x << ", " << p.y << ")\n";
        }
    }
}