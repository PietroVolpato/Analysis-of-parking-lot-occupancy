#include "ParkingSpaceClassifier.h"

using namespace cv;
using namespace std;

// Written by Ali Esmaeili nasab

ParkingSpaceClassifier::ParkingSpaceClassifier(double emptyThreshold) : emptyThreshold(emptyThreshold) {}

Mat ParkingSpaceClassifier::createBoundingBox(const Mat &parkingLotImage, const RotatedRect &rotated_rect) {
    Mat rotation_matrix = getRotationMatrix2D(rotated_rect.center, rotated_rect.angle, 1.0);
    Mat rotated_image;
    warpAffine(parkingLotImage, rotated_image, rotation_matrix, parkingLotImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
    
    Size rect_size = rotated_rect.size;
    cv::Rect bounding_box(
        static_cast<int>(rotated_rect.center.x - rect_size.width / 2),
        static_cast<int>(rotated_rect.center.y - rect_size.height / 2),
        static_cast<int>(rect_size.width),
        static_cast<int>(rect_size.height)
    );

    bounding_box &= cv::Rect(0, 0, rotated_image.cols, rotated_image.rows);
    Mat cropped_image;
    if (bounding_box.width > 0 && bounding_box.height > 0) {
        cropped_image = rotated_image(bounding_box);
    }
    return cropped_image;
}

Mat ParkingSpaceClassifier::preProcessing(const Mat &parkingLotImage, const Mat &parkingLotEmpty){
    Mat img_gray, img_blur, img_thresh, img_edge;
    cvtColor(parkingLotImage, img_gray, COLOR_BGR2GRAY);
    GaussianBlur(img_gray, img_blur, Size(5, 5), 0);
    Canny(img_blur, img_edge, 50, 150);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat dilated;
    dilate(img_edge, dilated, kernel);
    adaptiveThreshold(img_blur, img_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 16);
    
    Mat hsv, channels[3], edges, dilated2;
    cvtColor(parkingLotImage, hsv, COLOR_BGR2HSV);
    split(hsv, channels);
    Canny(channels[1], edges, 100, 150);
    dilate(edges, dilated2, kernel);
    
    Mat result;
    bitwise_or(dilated, img_thresh, result);
    
    Mat hsv_emp, channels_emp[3], edges_emp, dilated2_emp;
    cvtColor(parkingLotEmpty, hsv_emp, COLOR_BGR2HSV);
    split(hsv_emp, channels_emp);
    Canny(channels_emp[1], edges_emp, 100, 200);
    dilate(edges_emp, dilated2_emp, kernel, Point(-1, -1), 2);
    
    Mat subtracted;
    subtract(result, dilated2_emp, subtracted);
    
    std::vector<std::vector<Point>> contours;
    findContours(subtracted, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat filledImage = Mat::zeros(result.size(), CV_8UC1);
    drawContours(filledImage, contours, -1, Scalar(255), FILLED);
    
    Mat morph_final;
    morphologyEx(filledImage, morph_final, MORPH_CLOSE, kernel);

    return filledImage;
}

Mat ParkingSpaceClassifier::createSmallerBoundingBox(const Mat &parkingLotImage, const RotatedRect &rotated_rect) {
    RotatedRect smaller_rect(rotated_rect.center, Size2f(rotated_rect.size.width / 2, rotated_rect.size.height / 2), rotated_rect.angle);
    return createBoundingBox(parkingLotImage, smaller_rect);
}

void ParkingSpaceClassifier::classifyParkingSpaces(const Mat &parkingLotImage, const Mat &parkingLotEmpty, 
                                                   std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {
    Mat classification_input = preProcessing(parkingLotImage, parkingLotEmpty);
    
    occupancyStatus.clear();
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        Mat roi = createBoundingBox(classification_input, parkingSpaces[i]);
        int full = roi.rows * roi.cols;
        int count = countNonZero(roi);
        double ratio = static_cast<double>(count) / full;
        occupancyStatus.push_back(ratio >= emptyThreshold);
    }
}

void ParkingSpaceClassifier::calculateMetrics(const vector<bool>& original, const vector<bool>& predicted) {
    // Initialize counts for TP, FN, FP, TN
    int TP = 0, FN = 0, FP = 0, TN = 0;

    // Iterate through the vectors to calculate TP, FN, FP, TN
    for (size_t i = 0; i < original.size(); ++i) {
        if (original[i] && predicted[i]) {
            TP++; // True Positive
        } else if (original[i] && !predicted[i]) {
            FN++; // False Negative
        } else if (!original[i] && predicted[i]) {
            FP++; // False Positive
        } else {
            TN++; // True Negative
        }
    }

    // Calculate Precision, Recall, and Accuracy
    double precision = (TP + FP == 0) ? 0.0 : static_cast<double>(TP) / (TP + FP);
    double recall = (TP + FN == 0) ? 0.0 : static_cast<double>(TP) / (TP + FN);
    double accuracy = static_cast<double>(TP + TN) / (TP + TN + FP + FN);

    // Output the metrics
    std::cout << "TP: " << TP << std::endl;
    std::cout << "FN: " << FN << std::endl;
    std::cout << "FP: " << FP << std::endl;
    std::cout << "TN: " << TN << std::endl;
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "Accuracy: " << accuracy << std::endl;
}