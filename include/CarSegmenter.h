// PIETRO VOLPATO

#ifndef CAR_SEGMENTER_H
#define CAR_SEGMENTER_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

class CarSegmenter {
    private:
    cv::Ptr<cv::BackgroundSubtractor> bg;

    cv::Mat gammaCorrection (const cv::Mat& img, const double gamma);
    cv::Mat convertToGrayscale(const cv::Mat& img);
    cv::Mat equalization (const cv::Mat& img);
    cv::Mat applyRoi(const cv::Mat& img);
    float rotatedRectIntersectionArea(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);
    bool isCarInParking (const cv::RotatedRect& car, const std::vector<cv::RotatedRect>& parkingSpaces);

    public:
        CarSegmenter() {
            bg = cv::createBackgroundSubtractorMOG2();
        }

        cv::Mat preprocessImage(const cv::Mat& img, cv::String type);
        std::vector<std::vector<cv::Point>> findContoursSimple(const cv::Mat& img);
        void drawContourSimple (cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours);
        std::vector<cv::RotatedRect> findBBoxes(const std::vector<std::vector<cv::Point>>& contours);
        std::vector<cv::RotatedRect> filterBBoxes(const std::vector<cv::RotatedRect>& bboxes);
        void drawBBoxes(cv::Mat& img, const std::vector<cv::RotatedRect>& bboxes);
        void showImages(const cv::Mat& img);
        void trainBg(const std::vector<cv::Mat>& trainingVector);
        cv::Mat applyBg(const cv::Mat& img);
        cv::Mat enhanceMask(const cv::Mat& mask);
        cv::Mat segmentCar (const std::vector<cv::RotatedRect>& bboxes, const std::vector<cv::RotatedRect>& groundtruthBBoxes, const cv::Mat& mask, cv::Mat& img);
        cv::Mat createSegmentMask (const cv::Mat& img);
        cv::Mat applyMaskToImage (const cv::Mat& img, const cv::Mat& mask);
        std::vector<cv::Rect> newmethod(cv::Mat& mask, cv::Mat& img);
};

#endif