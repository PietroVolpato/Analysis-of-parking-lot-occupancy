#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <vector>

// Written by Ali Esmaeili nasab
class Visualizer {
public:
    Visualizer(int width, int height, std::vector<cv::RotatedRect>& parkingSpaces);

    cv::Mat createMockMinimap();
    void drawParkingSpaces(const cv::Mat &image, const std::vector<bool> &occupancyStatus);
    cv::Mat overlaySmallOnLarge(const cv::Mat& parkingSapaceClassified, const cv::Mat& visual2D);
    cv::Mat updateMinimap(const std::vector<bool>& occupancyStatus);

private:
    int minimap_width;
    int minimap_height;
    float spaceWidth  = 40.0f;
    float spaceHeight = 20.0f;
    float angle = -45.0f;
    float deltaX = spaceHeight / static_cast<float>(cos(angle * CV_PI / 180.0)); 
    float deltaY = (spaceWidth + spaceHeight) * static_cast<float>(cos(angle * CV_PI / 180.0)) ;
    float startX = static_cast<float>(minimap_width * 0.5 + deltaX * 5 + 1); // 5 parking space to the right from the middle of the image
    float startY = 60.0f;

    std::vector<int> sort_indices;         // Indices of Bboxes after sorting to be used for occupency
    std::vector<cv::RotatedRect> rectangles;
    std::vector<cv::RotatedRect> transformedRectangles;  // RotatedRects bassed on transformedCenters
    std::vector<cv::Point2f> originalCenters;
    std::vector<cv::Point2f> transformedCenters;        // Store te center after applting Honography
    std::vector<std::vector<cv::Point2f>> clusteredCenters;    // Clustered results

    cv::Mat minimap;

    void drawRotatedRect(cv::Mat &image, cv::Point2f center, float angle, cv::Scalar color);
    void rectParkingRow();
    std::vector<cv::Point2f> findExtremePoints();
    cv::Mat computeHomography(const std::vector<cv::Point2f>& srcPoints);
    std::vector<cv::Point2f> applyHomography(const cv::Mat& H);

    void clusterParkingSpaces(); // Method to cluster rectangles
};

#endif // VISUALIZER_H

