#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <vector>

class Visualizer {
public:
    Visualizer();  // Constructor

    cv::Mat createMockMinimap(int width, int height);
    void drawParkingSpaces(cv::Mat &image, const std::vector<cv::RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus);
private:
    float spaceWidth  = 40.0f;
    float spaceHeight = 20.0f;
    float angle = -45.0f;
    float startX = 400.0f;
    float startY = 60.0f;
    float deltaX = -spaceHeight / static_cast<float>(cos(angle * CV_PI / 180.0)); 
    float deltaY = (spaceWidth + spaceHeight) * static_cast<float>(cos(angle * CV_PI / 180.0));
    void drawRotatedRect(cv::Mat &image, const float &center_x, const float &center_y, const cv::Scalar &color, const bool &occupancyStatus);
    
};

#endif // VISUALIZER_H

