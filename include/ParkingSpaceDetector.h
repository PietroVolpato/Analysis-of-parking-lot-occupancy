#ifndef PARKING_SPACE_DETECTOR_H
#define PARKING_SPACE_DETECTOR_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cmath>


struct LineParams {
    double rho;
    double theta;
    cv::Vec4i endpoints;
};

std::vector<cv::Mat> loadImages (const int);

std::vector<LineParams> computeLineParams(const std::vector<cv::Vec4i>&);

std::vector<LineParams> computeLineParams(const std::vector<cv::Vec4i>&);

cv::Mat preprocessImage (const cv::Mat&);

void showImage (const cv::Mat&);

std::vector<cv::Vec4i> detectLines (const cv::Mat&, int, double, double);

void drawLines (cv::Mat&, const std::vector<cv::Vec4i>&);

std::vector<std::vector<LineParams>> clusterLinesByTheta(const std::vector<LineParams>&, double);

std::vector<cv::RotatedRect> detectParkingSpaces(const std::vector<cv::Vec4i>&);

cv::Mat drawParkingSpaces(const cv::Mat&, const std::vector<cv::RotatedRect>&);

#endif