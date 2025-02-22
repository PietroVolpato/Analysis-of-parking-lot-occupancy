// PIETRO VOLPATO

#ifndef PARKING_SPACE_DETECTOR_H
#define PARKING_SPACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

struct LineParams {
    double angle;
    cv::Vec4i endpoints;
    double length;
};

class ParkingSpaceDetector {
public:
    ParkingSpaceDetector() = default;

    std::vector<LineParams> computeLineParams(const std::vector<cv::Vec4i>& lines);
    cv::Mat preprocessImage(const cv::Mat& img);
    cv::Mat detectEdges(const cv::Mat& img, int threshold1, int threshold2, int aperture);
    void showImage(const cv::Mat& img);
    std::vector<cv::Vec4i> detectLines(const cv::Mat& img);
    void drawLines(cv::Mat& img, const std::vector<LineParams>& lines);
    std::vector<LineParams> filterLines(std::vector<LineParams>& lines);
    std::pair<std::vector<std::pair<LineParams, LineParams>>, std::vector<std::pair<LineParams, LineParams>>> clusterLines(const std::vector<LineParams>& lines);
    std::vector<cv::RotatedRect> detectParkingSpaces(const std::pair<std::vector<std::pair<LineParams, LineParams>>, std::vector<std::pair<LineParams, LineParams>>>& clusteredLines);
    std::vector<cv::RotatedRect> detectParkingSpacesSimple(const std::vector<LineParams>& lines);
    cv::Mat drawParkingSpaces(const cv::Mat& img, const std::vector<cv::RotatedRect>& parkingSpaces);

private:
    cv::Mat applyRoi(const cv::Mat& img);
    cv::Mat equalization(const cv::Mat& img);
    cv::Mat gammaCorrection(const cv::Mat& img, const double gamma);
    double distanceBetweenLines(const cv::Vec4i& line1, const cv::Vec4i& line2);
    // cv::Vec4i mergeTwoLines(const cv::Vec4i line1, const cv::Vec4i& line2);
    cv::Vec4i mergeCloseLines(const std::vector<cv::Vec4i>& lines);
    std::vector<cv::Vec4i> findCloseLines(const cv::Vec4i& reference, const std::vector<LineParams>& lines);
    bool isParallel(const double theta1, const double theta2);
    std::vector<cv::Vec4i> mergeLines(const std::vector<LineParams>& lines);
    cv::RotatedRect createBoundingBox(const LineParams& line1, const LineParams& line2);
};

#endif // PARKING_SPACE_DETECTOR_H