#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <vector>

class Visualizer {
public:
    Visualizer(int width = 400, int height = 300); // Default values , Constructor

    cv::Mat createMockMinimap(std::vector<bool> &occupancy);
    void drawParkingSpaces(cv::Mat &image, const std::vector<cv::RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus);

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
    void drawRotatedRect(cv::Mat &image, float center_x, float center_y, float angle, cv::Scalar color);
    void drawParkingRow(cv::Mat& image, int numSpaces, float startX, float startY, float angle, std::vector<bool>& occupancy, int& index);
    
};

#endif // VISUALIZER_H

