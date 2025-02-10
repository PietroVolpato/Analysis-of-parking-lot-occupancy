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

void classifyParkingSpaces(const Mat &parkingLotImage, const Mat &parkingLotEmpty, 
                           std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {
    double empty = 0.2;
    // Frame processing
    cv::Mat img_gray, img_stretched, img_clahe, img_blur, img_thresh;

    cv::cvtColor(parkingLotImage, img_gray, COLOR_BGR2GRAY);
    contrastStretching(img_gray, img_stretched);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(img_stretched, img_clahe);
    cv::GaussianBlur(img_clahe, img_blur, Size(3, 3), 1);
    cv::adaptiveThreshold(img_blur, img_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 16);
    // Blurring the image to reduce noise and normalize pixel value gaps caused by adaptive threshold
    cv::Mat blur;
    cv::medianBlur(img_thresh, blur, 5);

    // Dilating to increase foreground object
    Mat kernel_size = cv::Mat::ones(3, 3, CV_8U);
    cv::Mat dilate;
    cv::dilate(blur, dilate, kernel_size, Point(-1, -1), 1);


    occupancyStatus.clear();

    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        std::cout << "Roi #" << i << std::endl;
        const RotatedRect &rotated_rect = parkingSpaces[i];
        Mat roi = createBoundingBox(img_thresh, rotated_rect);

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


        // Classify: If too many foreground pixels, it's occupied
        occupancyStatus.push_back(occupied);
    }
}

// Function to draw the parking spaces on the image
void drawParkingSpaces(Mat &image, const std::vector<RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        Point2f vertices[4];
        parkingSpaces[i].points(vertices);
        
        Scalar color = occupancyStatus[i] ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        for (int j = 0; j < 4; j++)
            line(image, vertices[j], vertices[(j+1)%4], color, 2);
    }
}

void contrastStretching(cv::Mat& input, cv::Mat& output) {
    output = input.clone();  // Clone the input to output
    // Define the (r1, s1) and (r2, s2) points for contrast stretching
    int r1 = 70, s1 = 30;
    int r2 = 170, s2 = 220;
    int L = 256;  // Assuming 8-bit image (0 to 255 intensity levels)

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            int r = input.at<uchar>(i, j);
            int s = 0;

            // Apply the piecewise linear transformation
            if (r <= r1) {
                s = (s1 / (float)r1) * r;
            } else if (r <= r2) {
                s = ((s2 - s1) / (float)(r2 - r1)) * (r - r1) + s1;
            } else {
                s = ((L - 1 - s2) / (float)(L - 1 - r2)) * (r - r2) + s2;
            }

            output.at<uchar>(i, j) = cv::saturate_cast<uchar>(s);
        }
    }
}

