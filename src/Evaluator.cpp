#include "Evaluator.h"

// ---------------------------- mAP Implementation ----------------------------

// Compute Intersection over Union (IoU) for bounding boxes
double mAP::computeIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = box1.area() + box2.area() - intersectionArea;

    return unionArea > 0 ? static_cast<double>(intersectionArea) / unionArea : 0.0;
}

// Compute Average Precision (AP) for a given set of detections and ground truths
double mAP::computeAP(std::vector<Detection>& detections, const std::vector<cv::Rect>& groundTruths, double iouThreshold) {
    // Sort detections by descending confidence score
    std::sort(detections.begin(), detections.end(), [](const Detection &a, const Detection &b) {
        return a.score > b.score;
    });

    std::vector<bool> gtMatched(groundTruths.size(), false);
    std::vector<int> truePositives(detections.size(), 0);
    std::vector<int> falsePositives(detections.size(), 0);

    for (size_t i = 0; i < detections.size(); ++i) {
        double bestIoU = 0.0;
        int bestGtIdx = -1;
        for (size_t j = 0; j < groundTruths.size(); ++j) {
            double iou = computeIoU(detections[i].box, groundTruths[j]);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestGtIdx = j;
            }
        }
        if (bestIoU >= iouThreshold && bestGtIdx != -1 && !gtMatched[bestGtIdx]) {
            truePositives[i] = 1;
            gtMatched[bestGtIdx] = true;
        } else {
            falsePositives[i] = 1;
        }
    }

    std::vector<double> cumTP(detections.size(), 0);
    std::vector<double> cumFP(detections.size(), 0);
    for (size_t i = 0; i < detections.size(); ++i) {
        cumTP[i] = truePositives[i] + (i > 0 ? cumTP[i - 1] : 0);
        cumFP[i] = falsePositives[i] + (i > 0 ? cumFP[i - 1] : 0);
    }

    std::vector<double> precision(detections.size(), 0.0);
    std::vector<double> recall(detections.size(), 0.0);
    int totalGroundTruths = groundTruths.size();
    for (size_t i = 0; i < detections.size(); ++i) {
        precision[i] = cumTP[i] / (cumTP[i] + cumFP[i] + 1e-6);
        recall[i] = cumTP[i] / (totalGroundTruths + 1e-6);
    }

    double ap = 0.0;
    for (double t = 0.0; t <= 1.0; t += 0.1) {
        double p = 0.0;
        for (size_t i = 0; i < detections.size(); ++i) {
            if (recall[i] >= t) {
                p = std::max(p, precision[i]);
            }
        }
        ap += p / 11.0;
    }
    return ap;
}

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
