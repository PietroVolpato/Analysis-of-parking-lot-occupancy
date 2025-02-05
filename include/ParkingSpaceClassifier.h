#ifndef PARKINGSPACE_H
#define PARKINGSPACE_H

#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
cv::Mat createBoundingBox(const cv::Mat &parkingLotImage, cv::RotatedRect &rotated_rect);
void classifyParkingSpaces(const cv::Mat &parkingLotImage, const cv::Mat &parkingLotEmpty, std::vector<cv::RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus);
void drawParkingSpaces(cv::Mat &image, const std::vector<cv::RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus);
void contrastStretching(cv::Mat& input, cv::Mat& output);

#endif