// PIETRO VOLPATO
#ifndef LOADER_H
#define LOADER_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "tinyxml2.h"
#include <cmath>

class Loader {
    public:
        std::vector<cv::Mat> loadImagesFromSequence(const int sequence);
        std::vector<cv::Mat> loadImagesFromPath (cv::String path);
        std::vector<cv::RotatedRect> extractBoundingBoxesFromXML(const std::string &xmlFilePath, std::vector<bool> &occupancyStatus);
        std::vector<cv::String> loadXmlAddress(int sequence);
        std::vector<cv::Mat> loadMask (const int sequence);
};


#endif // LOADER_H