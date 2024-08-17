#ifndef PARKINGSPACE_H
#define PARKINGSPACE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "tinyxml2.h"
#include <vector>

cv::RotatedRect createBoundingBox(const cv::Point2f& center, const cv::Size2f& size, float angle);
bool isOccupied(const cv::Mat &roi);
void classifyParkingSpaces(const cv::Mat &parkingLotImage, std::vector<cv::RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus);
void drawParkingSpaces(cv::Mat &image, const std::vector<cv::RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus);

#endif