#ifndef PARKING_SPACE_DETECTOR_H
#define PARKING_SPACE_DETECTOR_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/ximgproc.hpp>
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

    std::vector<cv::Mat> loadImages(const int sequence);
    std::vector<LineParams> computeLineParams(const std::vector<cv::Vec4i>& lines);
    cv::Mat preprocessImage(const cv::Mat& img);
    cv::Mat detectEdges(const cv::Mat& img, int threshold1, int threshold2, int aperture);
    void showImage(const cv::Mat& img);
    std::vector<cv::Vec4i> detectLines(const cv::Mat& img, int threshold, double minLineLength, double maxLineGap);
    void drawLines(cv::Mat& img, const std::vector<LineParams>& lines);
    std::vector<LineParams> filterLines(std::vector<LineParams>& lines);
    std::pair<std::vector<LineParams>, std::vector<LineParams>> clusterLines(const std::vector<LineParams>& lines);
    std::vector<cv::RotatedRect> detectParkingSpaces(const std::pair<std::vector<LineParams>, std::vector<LineParams>>& clusteredLines);
    cv::Mat drawParkingSpaces(const cv::Mat& img, const std::vector<cv::RotatedRect>& parkingSpaces);

private:
    cv::Mat applyRoi(const cv::Mat& img);
    cv::Mat equalization(const cv::Mat& img);
    double distanceBetweenLines(const cv::Vec4i& line1, const cv::Vec4i& line2);
    // cv::Vec4i mergeTwoLines(const cv::Vec4i line1, const cv::Vec4i& line2);
    cv::Vec4i mergeCloseLines(const std::vector<cv::Vec4i>& lines);
    std::vector<cv::Vec4i> findCloseLines(const cv::Vec4i& reference, const std::vector<LineParams>& lines);
    bool isParallel(const double theta1, const double theta2);
    std::vector<cv::Vec4i> mergeLines(const std::vector<LineParams>& lines);
    cv::RotatedRect createBoundingBox(const LineParams& line1, const LineParams& line2);
};

#endif