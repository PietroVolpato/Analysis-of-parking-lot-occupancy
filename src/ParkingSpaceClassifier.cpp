#include "ParkingSpaceClassifier.h"

using namespace cv;

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

void classifyParkingSpaces(const Mat &parkingLotImage, const Mat &parkingLotEmpty, std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {
    // Step 1: Convert to grayscale before applying equalizeHist
    cv::Mat emptyGray, currentGray;
    cv::cvtColor(parkingLotEmpty, emptyGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(parkingLotImage, currentGray, cv::COLOR_BGR2GRAY);

    // Step 2: Apply Histogram Equalization
    cv::Mat emptyEqualized, currentEqualized;
    cv::equalizeHist(emptyGray, emptyEqualized);
    cv::equalizeHist(currentGray, currentEqualized);

    // Step 3: Compute absolute difference
    cv::Mat diff, thresh;
    cv::absdiff(emptyEqualized, currentEqualized, diff);

    // Step 4: Threshold to detect significant differences
    cv::threshold(diff, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    imshow("difference", thresh);
    waitKey(0);

    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        // Get the rotation matrix for the bounding box
        const RotatedRect &rotated_rect = parkingSpaces[i];
        
        Mat cropped_bbox = createBoundingBox(thresh, rotated_rect);          
        
        // Step 5: Calculate the percentage of white pixels
        int whitePixels = cv::countNonZero(cropped_bbox);  // Count white pixels
        double whiteRatio = (double)whitePixels / cropped_bbox.total();  // Ratio of white pixels

        std::cerr << whiteRatio << std::endl;
        
        // Heuristic: If there are many non-zero pixels, the space is occupied; otherwise, it's empty
        if (whiteRatio > 0.40) {  
            occupancyStatus[i] = true;
        } else {
            occupancyStatus[i] = false;
        }
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
