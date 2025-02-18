#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rrect, const cv::Scalar& color);
void drawParkingSpaces(cv::Mat &image, const std::vector<cv::RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus);
cv::Mat createMockMinimap(int width, int height);

#endif
