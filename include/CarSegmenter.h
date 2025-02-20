#ifndef CAR_SEGMENTER_H
#define CAR_SEGMENTER_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

class CarSegmenter {
    public:
        std::vector<cv::Mat> loadImages(const int sequence);
        cv::Mat createAvgImg (const std::vector<cv::Mat>& imgVector);
        cv::Mat preprocessImage(const cv::Mat& img, cv::String type);
        cv::Mat differenceImage(const cv::Mat& empty, const cv::Mat& img);
        cv::Mat analyzeImage(const cv::Mat& img);
        std::pair<std::vector<std::vector<cv::Point>>, std::vector<cv::Vec4i>> findContoursImg(const cv::Mat& img);
        void drawContoursImg(cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours, const std::vector<cv::Vec4i>& hierarchy);
        void showImages(const cv::Mat& img);

    private:
        cv::Mat gammaCorrection (const cv::Mat& img, const double gamma);
        cv::Mat convertToGrayscale(const cv::Mat& img);
        cv::Mat equalization (const cv::Mat& img);
};

#endif