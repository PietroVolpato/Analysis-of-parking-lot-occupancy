#ifndef PARKING_SPACE_DETECTOR_H
#define PARKING_SPACE_DETECTOR_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cmath>

std::vector<cv::Mat> loadImages (int);

std::vector<cv::Mat> preprocessImages (const std::vector<cv::Mat>&, int, double, double);

void showImages (const std::vector<cv::Mat>&);

std::vector<cv::Mat> detectEdges (const std::vector<cv::Mat>&, double, double);

std::vector<std::vector<cv::Vec4i>> detectLines (const std::vector<cv::Mat>&, int, double, double);

std::vector<std::vector<cv::Vec4i>> filterLines (const std::vector<std::vector<cv::Vec4i>>&);

std::vector<std::vector<cv::Vec4i>> clusterLines (const std::vector<std::vector<cv::Vec4i>>&);

void drawLines (std::vector<cv::Mat>&, const std::vector<std::vector<cv::Vec4i>>&);

void drawBoundingBoxes (std::vector<cv::Mat>&, const std::vector<std::vector<cv::Vec4i>>&);

#endif