#ifndef PARKINGSPACECLASSIFIER_H
#define PARKINGSPACECLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class ParkingSpaceClassifier {
public:
    // Constructor with an optional threshold parameter
    ParkingSpaceClassifier(double emptyThreshold = 0.4);
    
    // Constructer to classify parking spaces based on occupancy
    void classifyParkingSpaces(const cv::Mat &parkingLotImage, const cv::Mat &parkingLotEmpty, 
                               std::vector<cv::RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus);

private:
    double emptyThreshold; // Threshold for determining occupancy

    // Functions to create bounding boxes for parking spaces
    cv::Mat createBoundingBox(const cv::Mat &parkingLotImage, const cv::RotatedRect &rotated_rect);
    cv::Mat createSmallerBoundingBox(const cv::Mat &parkingLotImage, const cv::RotatedRect &rotated_rect);

    // Preprocessing the parking space image, the output is the binary image to be considered for occuency classiction
    cv::Mat preProcessing(const cv::Mat &parkingLotImage, const cv::Mat &parkingLotEmpty);
};

#endif // PARKINGSPACECLASSIFIER_H
