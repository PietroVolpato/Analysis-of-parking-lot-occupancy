#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <vector>

class Visualizer {
public:
    Visualizer(int width, int height, std::vector<cv::RotatedRect>& parkingSpaces); // Default values , Constructor

    cv::Mat createMockMinimap(std::vector<bool> &occupancy);
    void drawParkingSpaces(cv::Mat &image, const std::vector<bool> &occupancyStatus);
    cv::Mat overlaySmallOnLarge(const cv::Mat& parkingSapaceClassified, const cv::Mat& visual2D);

private:
    int minimap_width = 400;
    int minimap_height = 300;
    float spaceWidth  = 40.0f;
    float spaceHeight = 20.0f;
    float angle = -45.0f;
    float deltaX = spaceHeight / static_cast<float>(cos(angle * CV_PI / 180.0)); 
    float deltaY = (spaceWidth + spaceHeight) * static_cast<float>(cos(angle * CV_PI / 180.0));
    float startX = static_cast<float>(minimap_width * 0.5 + deltaX * 5); // 5 parking space to the right from the middle of the image
    float startY = 60.0f;
    std::vector<cv::RotatedRect> rectangles;


    void drawRotatedRect(cv::Mat &image, cv::Point2f center, float angle, cv::Scalar color);
    void drawParkingRow(cv::Mat& image, const cv::Mat &H, std::vector<bool>& occupancy, int& index);
    std::vector<cv::Point2f> findExtremePoints();
    cv::Mat computeHomography(const std::vector<cv::Point2f>& srcPoints);
    std::vector<cv::Point2f> applyHomography(const cv::Mat& H);
};

#endif // VISUALIZER_H

