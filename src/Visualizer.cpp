#include "Visualizer.h"

using namespace std;
using namespace cv;

// Written by Ali Esmaeili nasab

Visualizer::Visualizer(int width, int height, vector<RotatedRect>& parkingSpaces, vector<bool>& occupancyStatus) 
    : minimap_width(width), minimap_height(height), rectangles(parkingSpaces) {
        
        for (const auto& rect : rectangles) {
            originalCenters.push_back(rect.center);
        }
        Mat H = computeHomography(findExtremePoints());
        transformedCenters = applyHomography(H);

        if (originalCenters.size() != occupancyStatus.size()) {
            std::cerr << "Error: Parking centers and occupancy vectors must have the same size!\n";
            return;
        }

        // Store paired data
        for (size_t i = 0; i < originalCenters.size(); i++) {
            centersWithOccupancy.emplace_back(transformedCenters[i], occupancyStatus[i]);
        }
        rectParkingRow();


    }



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

void Visualizer::rectParkingRow() {
    
    // Get transformed centers
    clusterParkingSpaces();
    
    
    int angle;
    int index =0;
    vector<float> avgY_row;
    for (const auto& cluster : clusteredCenters) {
        angle = (index == 2) ? 45 : -45;

        // Calculate average y-value for the row
        float avgY = 0.0f;
        for (const auto& p : cluster) {
            avgY += p.first.y;
        }

        avgY /= cluster.size();
        avgY_row.push_back(avgY);
        
        float cx, cy, startX_row;
        cy = (index == 3) ? (avgY_row[index-1] - deltaY) : avgY;
        startX_row = (index == 2) ? (startX + 10) : startX;

        for (int i = 0; i < cluster.size(); i++) {
            cx = startX_row - i * Visualizer::deltaX;
            RotatedRect rect = RotatedRect(Point2f(cx, cy), Size2f(spaceWidth, spaceHeight), angle);
            transformedRectangles.push_back(rect);
        }
        index++;
    }
}

Mat Visualizer::overlaySmallOnLarge(const Mat& parkingSapaceClassified, const Mat& visual2D) {

    // Clone the large image to keep the original unchanged
    Mat outputImage = parkingSapaceClassified.clone();

    // The size for the small image (1/3 width of the large image)
    int newWidth = parkingSapaceClassified.cols / 3;
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



Mat Visualizer::updateMinimap(const vector<bool>& occupancyStatus) {
    // Create a white minimap
    Mat minimap(Visualizer::minimap_height, Visualizer::minimap_width, CV_8UC3, Scalar(255, 255, 255));
    
    if (occupancyStatus.size() != transformedRectangles.size()) {
        cerr << "Error: Occupancy vector size mismatch!\n";
        return minimap;
    }
    
    for (size_t i = 0; i < transformedRectangles.size(); ++i) {

        Scalar color = occupancyStatus[sort_indices[i]] ? Scalar(0, 0, 255) : Scalar(0, 255, 0); // Red for occupied, Green for free
        drawRotatedRect(minimap, transformedRectangles[i].center, transformedRectangles[i].angle, color);
    }
    imshow("mini", minimap);

    return minimap;
}

Mat Visualizer::createMockMinimap() {
    Mat emptyminimap(Visualizer::minimap_height, Visualizer::minimap_width, CV_8UC3, Scalar(255, 255, 255));

    // Draw only rectangle borders
    for (const auto& rect : transformedRectangles) {
        Point2f vertices[4];
        rect.points(vertices);
        for (int j = 0; j < 4; j++) {
            line(emptyminimap, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 0), 1);
        }
    }

    imshow("Mock Minimap", emptyminimap);
    waitKey(0);

    return emptyminimap;
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
        {minimap_width - 20, minimap_height - 20},  // Bottom-left (mapped to new coordinate space)
        {50, minimap_height - 20},    // Top-left
        {minimap_width - 20 , 50},    // Bottom-right
        {50, 50}       // Top-right
    };

    return getPerspectiveTransform(srcPoints, dstPoints);
}

// Apply homography transformation to all bounding box centers
vector<Point2f> Visualizer::applyHomography(const Mat& H) {
    vector<Point2f> transformedCentersP;

    perspectiveTransform(originalCenters, transformedCentersP, H);
    return transformedCentersP;
}

void Visualizer::clusterParkingSpaces() {
    if (centersWithOccupancy.empty()) return;

    int K = 4; // Number of clusters (rows)
    int N = centersWithOccupancy.size();

    // Prepare data for k-means clustering
    cv::Mat data(N, 1, CV_32F);
    for (int i = 0; i < N; i++) {
        data.at<float>(i, 0) = centersWithOccupancy[i].first.y;
    }

    // Run k-means clustering
    cv::Mat labels, centersMat;
    cv::kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1.0),
               10, cv::KMEANS_PP_CENTERS, centersMat);

    // Resize vector for 4 clusters
    clusteredCenters.resize(K);

    // Assign centers & occupancy to clusters
    for (int i = 0; i < N; i++) {
        int clusterIdx = labels.at<int>(i, 0);
        clusteredCenters[clusterIdx].push_back(centersWithOccupancy[i]); // Keeps occupancy linked
    }

    // Sort clusters based on increasing y-values of their centers
    std::sort(clusteredCenters.begin(), clusteredCenters.end(),
              [](const std::vector<std::pair<cv::Point2f, bool>>& a, const std::vector<std::pair<cv::Point2f, bool>>& b) {
                  return a.front().first.y > b.front().first.y;
              });
    // Sort each cluster by x-value
    for(auto& cluster : clusteredCenters){
        sort(cluster.begin(), cluster.end(), [](const auto& a, const auto& b) {
            return a.first.x > b.first.x; 
        });
    }

    // Finding the index coorelated with the occupency vector
    for (const auto& cluster : clusteredCenters) {
        for (const auto& pointPair : cluster) {
            // Find the index of the point in transformedCenters where both x and y coordinates match
            auto it = find_if(transformedCenters.begin(), transformedCenters.end(),
                            [&](const Point2f& p) { return p.x == pointPair.first.x && p.y == pointPair.first.y; });

            if (it != transformedCenters.end()) {
                sort_indices.push_back(distance(transformedCenters.begin(), it));
            } else {
                sort_indices.push_back(-1); // If not found
            }
        }
    }    
}
