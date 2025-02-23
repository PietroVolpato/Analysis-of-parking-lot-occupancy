#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

// mIoU class for image segmentation evaluation
class mIoU {
public:
    static double computeIoUForClass(const cv::Mat& gt, const cv::Mat& pred, int classID);
    static double computeMeanIoU(const cv::Mat& gt, const cv::Mat& pred, const std::vector<int>& classIDs);
};

#endif // EVALUATOR_H
