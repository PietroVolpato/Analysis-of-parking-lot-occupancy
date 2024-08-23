#ifndef GROUNDTRUTH_H
#define GROUNDTRUTH_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "tinyxml2.h"

// New functions for XML parsing and drawing
std::vector<cv::RotatedRect> extractBoundingBoxesFromXML(const std::string &xmlFilePath, std::vector<bool> &occupancyStatus);

#endif