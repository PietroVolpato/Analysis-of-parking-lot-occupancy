#ifndef PARKINGSPACE_H
#define PARKINGSPACE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

cv::RotatedRect createBoundingBox(const cv::Point2f& center, const cv::Size2f& size, float angle);
void classifyParkingSpaces(const cv::Mat &parkingLotImage, std::vector<cv::RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus);
void drawParkingSpaces(cv::Mat &image, const std::vector<cv::RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus);

// New functions for XML parsing and drawing
std::vector<cv::RotatedRect> extractBoundingBoxesFromXML(const std::string &xmlFilePath, std::vector<bool> &occupancyStatus);
void drawTrueParkingSpaces(cv::Mat &image, const std::string &xmlFilePath);

#endif