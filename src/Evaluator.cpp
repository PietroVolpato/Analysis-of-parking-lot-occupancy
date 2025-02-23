#include "Evaluator.h"

// ---------------------------- mIoU Implementation ----------------------------

// Compute the IoU for a single class
double mIoU::computeIoUForClass(const cv::Mat& gt, const cv::Mat& pred, int classID) {
    cv::Mat gtMask = (gt == classID);
    cv::Mat predMask = (pred == classID);

    cv::Mat intersection;
    cv::bitwise_and(gtMask, predMask, intersection);

    double intersectionCount = cv::countNonZero(intersection);
    double gtCount = cv::countNonZero(gtMask);
    double predCount = cv::countNonZero(predMask);
    double unionCount = gtCount + predCount - intersectionCount;

    if (unionCount == 0)
        return 1.0;

    return intersectionCount / unionCount;
}

// Compute mean IoU (mIoU) across a set of classes
double mIoU::computeMeanIoU(const cv::Mat& gt, const cv::Mat& pred, const std::vector<int>& classIDs) {
    double sumIoU = 0.0;
    int validClasses = 0;

    for (int classID : classIDs) {
        cv::Mat gtMask = (gt == classID);
        cv::Mat predMask = (pred == classID);

        if (cv::countNonZero(gtMask) == 0 && cv::countNonZero(predMask) == 0)
            continue;

        double iou = computeIoUForClass(gt, pred, classID);
        sumIoU += iou;
        validClasses++;

        // std::cout << "IoU for class " << classID << " : " << iou << std::endl;
    }

    return validClasses > 0 ? sumIoU / validClasses : 0.0;
}
