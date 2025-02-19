#ifndef BBOXSORTER_H
#define BBOXSORTER_H

#include <opencv2/opencv.hpp>
#include <vector>

class BboxSorter {
public:
    // Constructor
    BboxSorter(std::vector<cv::RotatedRect> rects);

    // Function to sort bounding boxes row-wise
    std::vector<cv::RotatedRect> sort();

private:
    std::vector<cv::RotatedRect> rectangles;  // Stores input bounding boxes

    // Helper methods
    cv::RotatedRect findBottomLeft();
    cv::RotatedRect* findNextInRow(cv::RotatedRect& current, std::vector<cv::RotatedRect*>& remaining);
    cv::RotatedRect* findNextRowStart(cv::RotatedRect& current, std::vector<cv::RotatedRect*>& remaining);
};

#endif // BBOXSORTER_H
