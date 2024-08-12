#ifndef PARKING_SPACE_DETECTOR_H
#define PARKING_SPACE_DETECTOR_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>
#include <cmath>

std::vector<cv::Mat> loadImages (int);

void showImages (const std::vector<cv::Mat>&);

cv::Mat detectEdges (const cv::Mat&);

std::vector<cv::Vec4i> detectLines (const cv::Mat&);

std::vector<std::vector<cv::Point>> detectContours (const cv::Mat&);

double calculateAngle (const cv::Vec4i&);

std::vector<std::vector<cv::Vec4i>> filterLines (const std::vector<cv::Vec4i>&);

void drawParkingSpaces (const cv::Mat&, const std::vector<cv::Vec4i>&, const std::vector<std::vector<cv::Point>>&);

void drawBoundingBoxes (const cv::Mat&, const std::vector<std::vector<cv::Vec4i>>&, const std::vector<std::vector<cv::Point>>&);

#endif