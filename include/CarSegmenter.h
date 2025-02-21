#ifndef CAR_SEGMENTER_H
#define CAR_SEGMENTER_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/ximgproc.hpp>
#include <opencv2/bgsegm.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <queue>
#include <pugixml.hpp>

class CarSegmenter {
    private:
    cv::Ptr<cv::BackgroundSubtractor> bg;

    cv::Mat gammaCorrection (const cv::Mat& img, const double gamma);
    cv::Mat convertToGrayscale(const cv::Mat& img);
    cv::Mat equalization (const cv::Mat& img);
    float rotatedRectIntersectionArea(const cv::RotatedRect& rect1, const cv::RotatedRect& rect2);
    bool isCarInParking (const cv::RotatedRect& car, const std::vector<cv::RotatedRect>& parkingSpaces);

    public:
        CarSegmenter() {
            // bg = cv::createBackgroundSubtractorMOG();
            bg = cv::createBackgroundSubtractorMOG2();
        }

        std::vector<cv::Mat> loadImages(const int sequence);
        std::vector<cv::RotatedRect> getBBoxes (cv::String filePath);
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
};

#endif