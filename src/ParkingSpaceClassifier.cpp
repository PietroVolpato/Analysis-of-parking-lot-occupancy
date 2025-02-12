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
    double empty = 0.3;
    // Frame processing
    cv::Mat img_gray, img_stretched, img_clahe, img_blur, img_thresh, img_edge;

    //cv::cvtColor(parkingLotImage, img_gray, COLOR_BGR2GRAY);
    cv::cvtColor(parkingLotImage, img_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(img_gray, img_blur, cv::Size(5, 5), 0);
    cv::Canny(img_blur, img_edge, 50, 150);
    //cv::adaptiveThreshold(img_edge, img_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 16);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat dilated;
    cv::dilate(img_edge, dilated, kernel, cv::Point(-1, -1), 1);  // Expand regions
    //cv::morphologyEx(img_thresh, img_stretched, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    
    // edge image
    Mat hsv, channels[3], edges, dilated2;
    cvtColor(parkingLotImage, hsv, COLOR_BGR2HSV);
    split(hsv, channels);
    Canny(channels[1], edges, 100, 200);
    cv::dilate(edges, dilated2, kernel, cv::Point(-1, -1), 1);  // Expand regions

    //edge on empty
    Mat hsv_emp, channels_emp[3], edges_emp, dilated2_emp;
    cvtColor(parkingLotEmpty, hsv_emp, COLOR_BGR2HSV);
    split(hsv_emp, channels_emp);
    Canny(channels_emp[1], edges_emp, 100, 200);
    cv::dilate(edges_emp, dilated2_emp, kernel, cv::Point(-1, -1), 2);  // Expand regions


    // Perform bitwise OR
    cv::Mat result;
    cv::bitwise_or(dilated, dilated2, result);

    //subtracting parking lines
    Mat subtracted;
    cv::subtract(result, dilated2_emp, subtracted);

    Mat morph_final;
    morphologyEx(subtracted, morph_final, cv::MORPH_CLOSE, kernel);

    // Apply dilation to enlarge regions
    cv::Mat dilated_final;
    cv::erode(morph_final, dilated_final, kernel);

    // Apply closing (dilation followed by erosion) to connect regions
    cv::Mat closed;
    cv::morphologyEx(dilated_final, closed, cv::MORPH_CLOSE, kernel);

    //imshow("hsv", subtracted);
    //imshow("rgb", dilated);
    imshow("sum", closed);
    waitKey(0);


    //contrastStretching(mask, img_stretched);
    //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    //clahe->apply(img_stretched, img_clahe);
    // cv::GaussianBlur(img_stretched, img_blur, Size(5, 5), 0);
    // imshow("blur", img_blur);
    // waitKey(0);
    // cv::adaptiveThreshold(img_blur, img_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 16);

    occupancyStatus.clear();

    Mat classification_input = closed;

    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        std::cout << "Roi #" << i << std::endl;
        const RotatedRect &rotated_rect = parkingSpaces[i];
        Mat roi = createSmallerBoundingBox(classification_input, rotated_rect);
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

