#ifndef METRICS_H
#define METRICS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

// Structure for detections with confidence scores
struct Detection {
    cv::Rect box;
    double score;
};

// mAP class for object detection evaluation
class mAP {
public:
    static double computeIoU(const cv::Rect& box1, const cv::Rect& box2);
    static double computeAP(std::vector<Detection>& detections, const std::vector<cv::Rect>& groundTruths, double iouThreshold = 0.5);
};

// mIoU class for image segmentation evaluation
class mIoU {
public:
    static double computeIoUForClass(const cv::Mat& gt, const cv::Mat& pred, int classID);
    static double computeMeanIoU(const cv::Mat& gt, const cv::Mat& pred, const std::vector<int>& classIDs);
};

#endif // METRICS_H
