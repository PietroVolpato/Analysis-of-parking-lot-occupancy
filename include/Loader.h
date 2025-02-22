// PIETRO VOLPATO
#ifndef LOADER_H
#define LOADER_H

#include <pugixml.hpp>
#include <opencv2/opencv.hpp>

class Loader {
    public:
        std::vector<cv::Mat> loadImagesFromSequence(const int sequence);
        std::vector<cv::Mat> loadImagesFromPath (cv::String path);
        std::vector<cv::RotatedRect> getBBoxes (cv::String filePath);
        std::vector<cv::Mat> loadMask (const int sequence);
};


#endif // LOADER_H