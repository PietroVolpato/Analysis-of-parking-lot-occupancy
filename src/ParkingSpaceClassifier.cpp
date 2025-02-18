#include "ParkingSpaceClassifier.h"

using namespace cv;
using namespace std;

// Function to create a bounding box
Mat createBoundingBox(const Mat &parkingLotImage, const RotatedRect &rotated_rect) {
    Mat rotation_matrix = getRotationMatrix2D(rotated_rect.center, rotated_rect.angle, 1.0);
        // Rotate the entire image around the bounding box center
        Mat rotated_image;
        warpAffine(parkingLotImage, rotated_image, rotation_matrix, parkingLotImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
        
        // Extract the ROI from the rotated image
        Size rect_size = rotated_rect.size;
        
        // Create a 3x3 full rotation matrix from the 2x3 rotation matrix
        cv::Rect bounding_box(
            static_cast<int>(rotated_rect.center.x - rect_size.width / 2),
            static_cast<int>(rotated_rect.center.y - rect_size.height / 2),
            static_cast<int>(rect_size.width),
            static_cast<int>(rect_size.height)
        );

        // Ensure the bounding box is within the image bounds
        bounding_box &= cv::Rect(0, 0, rotated_image.cols, rotated_image.rows);
        Mat cropped_image;

        if (bounding_box.width > 0 && bounding_box.height > 0) {
            cropped_image = rotated_image(bounding_box);
        }
    return cropped_image;
}

Mat createSmallerBoundingBox(const Mat &parkingLotImage, const RotatedRect &rotated_rect) {
    // Scale down the rotated rectangle size by half
    RotatedRect smaller_rect(rotated_rect.center, 
                             Size2f(rotated_rect.size.width / 2, rotated_rect.size.height / 2), 
                             rotated_rect.angle);

    // Call the existing function with the smaller rectangle
    return createBoundingBox(parkingLotImage, smaller_rect);
}


void classifyParkingSpaces(const Mat &parkingLotImage, const Mat &parkingLotEmpty, 
                           std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {
    double empty = 0.4;
    // Frame processing
    cv::Mat img_gray, img_stretched, img_clahe, img_blur, img_thresh, img_edge;
    cv::cvtColor(parkingLotImage, img_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(img_gray, img_blur, cv::Size(5, 5), 0);

    // edge detection
    cv::Canny(img_blur, img_edge, 50, 150);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat dilated;
    cv::dilate(img_edge, dilated, kernel, cv::Point(-1, -1), 1);  // Expand regions
    
    // Thresholding
    adaptiveThreshold(img_blur, img_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 16);


    // edge image on HSV channel
    Mat hsv, channels[3], edges, dilated2;
    cvtColor(parkingLotImage, hsv, COLOR_BGR2HSV);
    split(hsv, channels);
    Canny(channels[1], edges, 100, 150);
    cv::dilate(edges, dilated2, kernel, cv::Point(-1, -1), 1);  // Expand regions


    // Perform bitwise OR
    cv::Mat result;
    cv::bitwise_or(dilated, img_thresh, result);

    //edge on empty
    Mat hsv_emp, channels_emp[3], edges_emp, dilated2_emp;
    cvtColor(parkingLotEmpty, hsv_emp, COLOR_BGR2HSV);
    split(hsv_emp, channels_emp);
    Canny(channels_emp[1], edges_emp, 100, 200);
    cv::dilate(edges_emp, dilated2_emp, kernel, cv::Point(-1, -1), 2);  // Expand regions

    //subtracting parking lines
    Mat subtracted;
    cv::subtract(result, dilated2_emp, subtracted);
    

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(subtracted, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // Fill the contours
    cv::Mat filledImage = cv::Mat::zeros(result.size(), CV_8UC1);
    cv::drawContours(filledImage, contours, -1, cv::Scalar(255), cv::FILLED);

    Mat morph_final;
    morphologyEx(filledImage, morph_final, cv::MORPH_CLOSE, kernel);

    imshow("filledImage", filledImage);
    waitKey(0);

    occupancyStatus.clear();

    Mat classification_input = filledImage;

    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        std::cout << "Roi #" << i << std::endl;
        const RotatedRect &rotated_rect = parkingSpaces[i];
        Mat roi = createBoundingBox(classification_input, rotated_rect);
        // imshow("roi", roi);
        // waitKey(0);
        int full = roi.rows * roi.cols;
        int count = countNonZero(roi);
        bool occupied;
        double ratio = static_cast<double>(count) / full;
        std::cout << ratio << std::endl; 
        if (ratio < empty) {
            occupied = false;
        } else {
            occupied = true;
        }


        occupancyStatus.push_back(occupied);
    }
}



