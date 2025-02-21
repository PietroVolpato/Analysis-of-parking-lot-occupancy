#ifndef BBOXSORTER_H
#define BBOXSORTER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>  // For std::iota
#include <algorithm>
#include <iostream>

class BboxSorter {
public:
    // Constructor
    explicit BboxSorter(std::vector<cv::RotatedRect> rects);

    // Function to sort bounding boxes using homography
    std::vector<cv::RotatedRect> sort();
    void drawTransformedCenters(const std::vector<cv::Point2f>& transformedCenters);

private:
    std::vector<cv::RotatedRect> rectangles;

    // Helper functions
    std::vector<cv::Point2f> findExtremePoints();
    cv::Mat computeHomography(const std::vector<cv::Point2f>& srcPoints);
    std::vector<cv::Point2f> applyHomography(const cv::Mat& H);
};

#endif // BBOXSORTER_H
